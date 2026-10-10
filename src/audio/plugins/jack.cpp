/**
 *  OSM
 *  Copyright (C) 2022  Adrian Schollmeyer

 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.

 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.

 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "jack.h"
#include <QCoreApplication>
#include <QDebug>
#include <QSocketNotifier>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <unistd.h>

namespace {
constexpr unsigned int DEFAULT_SAMPLE_RATE{48000};
constexpr unsigned int DEFAULT_CHANNEL_COUNT_INPUT{2};
constexpr unsigned int DEFAULT_CHANNEL_COUNT_OUTPUT{1};

constexpr const char* PLUGIN_NAME{"JACK"};
constexpr const char* JACK_CLIENT_NAME{"OpenSoundMeter"};
constexpr const char* DEFAULT_MEASUREMENT_CHANNEL{"measurement"};
constexpr const char* DEFAULT_REFERENCE_CHANNEL{"reference"};
constexpr const char* DEFAULT_OUTPUT_CHANNEL{"out"};

// Frames the JACK thread interleaves per step through a stack buffer.
constexpr std::size_t RT_CHUNK_FRAMES{256};
constexpr std::size_t MAX_CHANNELS{2};
// Frames the endpoint thread moves per QIODevice call.
constexpr std::size_t TRANSFER_FRAMES{4096};
// How often the endpoint thread polls the ring. Well below the input ring
// length and the output fill target, so neither runs dry between polls.
constexpr std::chrono::milliseconds ENDPOINT_POLL{2};

int terminationPipe[2]{-1, -1};

void onTerminationSignal(int signal)
{
    // Async-signal context: only wake the event loop. A second signal gets
    // the default action, so a hung process can still be stopped.
    std::signal(signal, SIG_DFL);
    const char c{1};
    [[maybe_unused]] auto res = ::write(terminationPipe[1], &c, 1);
}
}

namespace audio {
JackClient::JackClient(const Plugin::Direction &direction, QIODevice& endpoint)
    : m_client(nullptr, jackClientDeleter()), m_direction(direction), m_endpoint(endpoint),
      m_ring(nullptr, jack_ringbuffer_free)
{
    initJackClient();
    initPorts();
    initSampleRate();
    initBufferSize();
    initProcessing();
}

JackClient::~JackClient()
{
    // Explicitly stop processing first, to prevent us still running the JACK
    // processor while the rest of the object is being cleaned up
    if (m_client)
        close();
}

jack_nframes_t JackClient::currentSampleRate() const
{
    return m_sampleRate;
}

std::size_t JackClient::currentChannelCount() const
{
    return m_ports.size();
}

Plugin::Direction JackClient::direction() const
{
    return m_direction;
}

void JackClient::close()
{
    // jack_deactivate returns only after the last process cycle finished, so
    // the endpoint thread and the ring can go next, then ports and client.
    deactivate();
    stopEndpointThread();
    m_client.reset(nullptr);
}

void JackClient::initJackClient()
{
    jack_status_t jackStatus;
    jack_client_t* jackClient = jack_client_open(
        JACK_CLIENT_NAME,
        JackOptions::JackNoStartServer,
        &jackStatus
    );

    if (!jackClient)
        throw JackPluginException{"Failed to open JACK client"};

    m_client.reset(jackClient);
}

void JackClient::initSampleRate()
{
    assert(m_client);
    m_sampleRate = jack_get_sample_rate(m_client.get());
    qDebug() << "[JACK] Sample rate set: " << m_sampleRate;

    int res = jack_set_sample_rate_callback(
        m_client.get(),
        jackSampleRateCallback,
        this
    );
    if (res != 0)
        throw JackPluginException{"Failed to register sample rate callback"};
}

void JackClient::initBufferSize()
{
    assert(m_client);
    assert(!m_ports.empty());
    m_bufferSize = jack_get_buffer_size(m_client.get());
    qDebug() << "[JACK] Buffer size set: " << m_bufferSize;

    int res = jack_set_buffer_size_callback(
        m_client.get(),
        jackBufferSizeCallback,
        this
    );
    if (res != 0)
        throw JackPluginException{"Failed to register buffer size callback"};
}

void JackClient::initPorts()
{
    assert(m_client);
    switch (m_direction) {
    case Plugin::Direction::Input:
        registerInputs();
        break;

    case Plugin::Direction::Output:
        registerOutputs();
        break;

    default:
        throw JackPluginException{"Unsupported signal direction"};
    }
}

void JackClient::initProcessing()
{
    assert(m_client);
    int res = jack_set_process_callback(
        m_client.get(),
        jackProcessCallback,
        this
    );
    if (res != 0)
        throw JackPluginException{"Failed to register process callback"};

    const std::size_t frameBytes = m_ports.size() * sizeof(float);
    m_ring.reset(jack_ringbuffer_create(RING_SECONDS * m_sampleRate * frameBytes));
    if (!m_ring)
        throw JackPluginException{"Failed to allocate ring buffer"};
    jack_ringbuffer_mlock(m_ring.get());
    m_transferBuffer.resize(TRANSFER_FRAMES * m_ports.size());
}

void JackClient::activate()
{
    assert(m_client);
    if (m_active)
        return;

    qDebug() << "[JACK] Activating";
    if (m_direction == Plugin::Direction::Output)
        fillOutputRing();
    startEndpointThread();
    m_active = true;
    int res = jack_activate(m_client.get());
    if (res != 0) {
        m_active = false;
        stopEndpointThread();
        throw JackPluginException{"Failed to activate client"};
    }
}

void JackClient::deactivate()
{
    if (!m_active)
        return;

    m_active = false;

    assert(m_client);
    // Called from the destructor too, and fails when the server is gone:
    // the client must still be closed, so report instead of throwing.
    if (jack_deactivate(m_client.get()) != 0)
        qWarning() << "[JACK] Failed to deactivate client";
}

void JackClient::startEndpointThread()
{
    if (m_endpointRunning)
        return;
    m_endpointRunning = true;
    m_endpointThread = std::thread(&JackClient::endpointThreadLoop, this);
}

void JackClient::stopEndpointThread()
{
    m_endpointRunning = false;
    if (m_endpointThread.joinable())
        m_endpointThread.join();
}

std::size_t JackClient::outputTargetBytes() const
{
    // Enough queued output to ride out a few scheduling hiccups of the
    // endpoint thread, small enough to keep generator latency low.
    const std::size_t frames = std::max<std::size_t>(4 * m_bufferSize, m_sampleRate / 50);
    return frames * m_ports.size() * sizeof(float);
}

void JackClient::fillOutputRing()
{
    const std::size_t frameBytes = m_ports.size() * sizeof(float);
    const std::size_t target = outputTargetBytes();
    std::size_t queued = jack_ringbuffer_read_space(m_ring.get());
    while (queued < target && m_endpoint.isReadable()) {
        std::size_t bytes = std::min(target - queued, m_transferBuffer.size() * sizeof(float));
        bytes = std::min(bytes, jack_ringbuffer_write_space(m_ring.get()));
        bytes -= bytes % frameBytes;
        if (bytes == 0)
            break;
        char *data = reinterpret_cast<char *>(m_transferBuffer.data());
        qint64 got = m_endpoint.read(data, bytes);
        if (got <= 0)
            break;
        got -= got % frameBytes;
        jack_ringbuffer_write(m_ring.get(), data, got);
        queued += got;
    }
}

void JackClient::endpointThreadLoop()
{
    const std::size_t frameBytes = m_ports.size() * sizeof(float);
    char *data = reinterpret_cast<char *>(m_transferBuffer.data());
    unsigned int reportedDropouts = 0;
    while (m_endpointRunning) {
        if (m_direction == Plugin::Direction::Input) {
            std::size_t bytes;
            while ((bytes = std::min(jack_ringbuffer_read_space(m_ring.get()),
                                     m_transferBuffer.size() * sizeof(float))) >= frameBytes) {
                bytes -= bytes % frameBytes;
                jack_ringbuffer_read(m_ring.get(), data, bytes);
                if (m_endpoint.isWritable())
                    m_endpoint.write(data, bytes);
            }
        } else {
            fillOutputRing();
        }

        const unsigned int dropouts = m_dropouts;
        if (dropouts != reportedDropouts) {
            qWarning() << "[JACK]" << (m_direction == Plugin::Direction::Input ? "input overruns:" : "output underruns:")
                       << dropouts;
            reportedDropouts = dropouts;
        }
        std::this_thread::sleep_for(ENDPOINT_POLL);
    }
}

void JackClient::registerInputs()
{
    qDebug() << "[JACK] Registering Inputs";
    jack_port_t *measurementPort = jack_port_register(
        m_client.get(),
        DEFAULT_MEASUREMENT_CHANNEL,
        JACK_DEFAULT_AUDIO_TYPE,
        JackPortFlags::JackPortIsInput,
        0
    );
    if (!measurementPort)
        throw JackPluginException{"Failed to register measurement port"};
    m_ports.emplace_back(measurementPort, jackPortDeleter(m_client.get()));
    m_portBuffers.push_back(nullptr);

    jack_port_t *referencePort = jack_port_register(
        m_client.get(),
        DEFAULT_REFERENCE_CHANNEL,
        JACK_DEFAULT_AUDIO_TYPE,
        JackPortFlags::JackPortIsInput,
        0
    );
    if (!referencePort)
        throw JackPluginException{"Failed to register reference port"};
    m_ports.emplace_back(referencePort, jackPortDeleter(m_client.get()));
    m_portBuffers.push_back(nullptr);
}

void JackClient::registerOutputs()
{
    qDebug() << "[JACK] Registering Outputs";
    jack_port_t *outputPort = jack_port_register(
        m_client.get(),
        DEFAULT_OUTPUT_CHANNEL,
        JACK_DEFAULT_AUDIO_TYPE,
        JackPortFlags::JackPortIsOutput,
        0
    );
    if (!outputPort)
        throw JackPluginException{"Failed to register output port"};
    m_ports.emplace_back(outputPort, jackPortDeleter(m_client.get()));
    m_portBuffers.push_back(nullptr);
}

std::function<void(jack_client_t *)> JackClient::jackClientDeleter()
{
    return [this](jack_client_t *client) -> void {
        m_ports.resize(0);
        jack_client_close(client);
    };
}

std::function<void(jack_port_t *)> JackClient::jackPortDeleter(jack_client_t *client)
{
    // We can't use the object's client member here as it might already have
    // changed if this deleter is called in the client object's deleter,
    // resulting in an assertion failure or segfault.
    assert(client);
    return [client](jack_port_t *port) -> void {
        jack_port_unregister(client, port);
    };
}

int JackClient::jackSampleRateCallback(jack_nframes_t newSampleRate, void* obj)
{
    assert(obj);
    return reinterpret_cast<JackClient*>(obj)->jackSampleRateCallbackInt(newSampleRate);
}

int JackClient::jackSampleRateCallbackInt(jack_nframes_t newSampleRate)
{
    m_sampleRate = newSampleRate;
    qInfo() << "[JACK] Sample rate updated: " << m_sampleRate;
    emit sampleRateChanged(m_sampleRate);
    return 0;
}

int JackClient::jackBufferSizeCallback(jack_nframes_t newBufferSize, void* obj)
{
    assert(obj);
    return reinterpret_cast<JackClient*>(obj)->jackBufferSizeCallbackInt(newBufferSize);
}

int JackClient::jackBufferSizeCallbackInt(jack_nframes_t newBufferSize)
{
    m_bufferSize = newBufferSize;
    qInfo() << "[JACK] Buffer size updated: " << newBufferSize;
    return 0;
}

int JackClient::jackProcessCallback(jack_nframes_t nframes, void* obj)
{
    assert(obj);
    return reinterpret_cast<JackClient*>(obj)->jackProcessCallbackInt(nframes);
}

int JackClient::jackProcessCallbackInt(jack_nframes_t nframes)
{
    // Real-time thread: no locks, allocation, Qt calls or logging below.
    if (!m_active || nframes == 0)
        return 0;

    for (std::size_t i = 0; i < m_ports.size(); ++i) {
        m_portBuffers[i] = reinterpret_cast<float*>(
            jack_port_get_buffer(m_ports[i].get(), nframes));
    }

    if (m_direction == Plugin::Direction::Input) {
        processInputPorts(nframes);
    } else {
        processOutputPorts(nframes);
    }
    return 0;
}

void JackClient::processInputPorts(jack_nframes_t nframes)
{
    const std::size_t channels = m_portBuffers.size();
    if (jack_ringbuffer_write_space(m_ring.get()) < nframes * channels * sizeof(float)) {
        // Endpoint thread fell behind by the whole ring; drop this period
        // whole so channels stay aligned in the interleaved stream.
        m_dropouts.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    float chunk[RT_CHUNK_FRAMES * MAX_CHANNELS];
    for (jack_nframes_t start = 0; start < nframes; start += RT_CHUNK_FRAMES) {
        const jack_nframes_t frames = std::min<jack_nframes_t>(RT_CHUNK_FRAMES, nframes - start);
        std::size_t pos{0};
        for (jack_nframes_t i = start; i < start + frames; ++i) {
            for (const float* portBuf : m_portBuffers)
                chunk[pos++] = portBuf[i];
        }
        jack_ringbuffer_write(m_ring.get(), reinterpret_cast<const char*>(chunk), pos * sizeof(float));
    }
}

void JackClient::processOutputPorts(jack_nframes_t nframes)
{
    const std::size_t channels = m_portBuffers.size();
    if (jack_ringbuffer_read_space(m_ring.get()) < nframes * channels * sizeof(float)) {
        // Underrun: play silence rather than wait for the endpoint thread.
        for (float* portBuf : m_portBuffers)
            std::fill(portBuf, portBuf + nframes, 0.f);
        m_dropouts.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    float chunk[RT_CHUNK_FRAMES * MAX_CHANNELS];
    for (jack_nframes_t start = 0; start < nframes; start += RT_CHUNK_FRAMES) {
        const jack_nframes_t frames = std::min<jack_nframes_t>(RT_CHUNK_FRAMES, nframes - start);
        jack_ringbuffer_read(m_ring.get(), reinterpret_cast<char*>(chunk), frames * channels * sizeof(float));
        std::size_t pos{0};
        for (jack_nframes_t i = start; i < start + frames; ++i) {
            for (float* portBuf : m_portBuffers)
                portBuf[i] = chunk[pos++];
        }
    }
}

JackPlugin::JackPlugin()
{
    DeviceInfo m_defaultDeviceInfo{PLUGIN_NAME, PLUGIN_NAME};
    m_defaultDeviceInfo.setName(PLUGIN_NAME);
    m_defaultDeviceInfo.setInputChannels({
        DEFAULT_MEASUREMENT_CHANNEL,
        DEFAULT_REFERENCE_CHANNEL
    });
    m_defaultDeviceInfo.setOutputChannels({DEFAULT_OUTPUT_CHANNEL});
    m_defaultDeviceInfo.setDefaultSampleRate(DEFAULT_SAMPLE_RATE);
    m_list.push_back(m_defaultDeviceInfo);

    installTerminationHandler();
}

void JackPlugin::installTerminationHandler()
{
    // SIGTERM's default action kills the process with its clients still in
    // the JACK graph, and the server misses cycles waiting on dead clients.
    // Leave the graph cleanly first, then quit as a normal exit would.
    if (terminationPipe[0] >= 0 || ::pipe2(terminationPipe, O_CLOEXEC | O_NONBLOCK) != 0)
        return;

    auto *notifier = new QSocketNotifier(terminationPipe[0], QSocketNotifier::Read, this);
    connect(notifier, &QSocketNotifier::activated, this, [this]() {
        char c;
        while (::read(terminationPipe[0], &c, 1) > 0) {}
        qInfo() << "[JACK] Termination signal, closing JACK clients";
        m_clients.clear();
        QCoreApplication::quit();
    });

    struct sigaction action {};
    action.sa_handler = onTerminationSignal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGINT, &action, nullptr);
}

QString JackPlugin::name() const
{
    return PLUGIN_NAME;
}

DeviceInfo::List JackPlugin::getDeviceInfoList() const
{
    return m_list;
}

DeviceInfo::Id JackPlugin::defaultDeviceId(
    [[maybe_unused]] const Plugin::Direction &mode) const
{
    return DeviceInfo::Id();
}

Format JackPlugin::deviceFormat(
    [[maybe_unused]] const DeviceInfo::Id &id,
    const Plugin::Direction &mode) const
{
    const auto client_it = std::find_if(
        m_clients.begin(),
        m_clients.end(),
        [mode](const auto& client) {

        return client.second->direction() == mode;
    });

    if (client_it == m_clients.end()) {
        if (mode == Plugin::Direction::Input)
            return {DEFAULT_SAMPLE_RATE, DEFAULT_CHANNEL_COUNT_INPUT};
        else
            return {DEFAULT_SAMPLE_RATE, DEFAULT_CHANNEL_COUNT_OUTPUT};
    } else {
        return {
            static_cast<unsigned int>(client_it->second->currentSampleRate()),
            static_cast<unsigned int>(client_it->second->currentChannelCount())
        };
    }
}

Stream *JackPlugin::open(
    [[maybe_unused]] const DeviceInfo::Id &id,
    const Plugin::Direction &mode,
    [[maybe_unused]] const Format &format,
    QIODevice *endpoint)
{
    // The sample rate is set by JACK, so we don't mess with it.
    // The channel count is fixed to the minimum amount of channels necessary
    // for OSM to work as we can freely patch the cannels in JACK.
    //
    // The ID can be ignored as we create individual JACK clients
    // for each requested Stream.
    try {
        std::unique_ptr<JackClient> jack = std::make_unique<JackClient>(mode, *endpoint);

        Format f{
            jack->currentSampleRate(),
            static_cast<unsigned int>(jack->currentChannelCount())
        };
        Stream *stream = new Stream(f);

        m_clients.emplace(stream, std::move(jack));
        JackClient& jackClient = *m_clients.at(stream).get();

        connect(stream, &Stream::closeMe, this, [this, stream, endpoint]() {
            // Order is important here to prevent JACK still writing data to
            // the endpoint after it has been closed.
            m_clients.erase(stream);

            if (endpoint->isOpen())
                endpoint->close();

            stream->deleteLater();
        });

        connect(
            &jackClient,
            &JackClient::sampleRateChanged,
            this,
            [stream](jack_nframes_t newRate) {
                assert(stream);
                stream->setSampleRate(static_cast<unsigned int>(newRate));
        });

        endpoint->open(mode == Input ? QIODevice::WriteOnly : QIODevice::ReadOnly);
        jackClient.activate();

        return stream;
    } catch (JackPluginException& e) {
        return nullptr;
    }
}

}
