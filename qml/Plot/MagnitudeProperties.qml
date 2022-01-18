/**
 *  OSM
 *  Copyright (C) 2019  Pavel Smokotnin

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
import QtQuick 2.9
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.3
import QtQuick.Dialogs 1.3

import "../elements"
import SourceModel 1.0
import OpenSoundMeter 1.0

Item {
    id: chartProperties
    property var dataObject

    ColumnLayout {
        spacing: 0
        anchors.fill: parent

        RowLayout {
            spacing: 0

            SelectableSpinBox {
                value: dataObject.xmin
                onValueChanged: dataObject.xmin = value
                from: dataObject.xLowLimit
                to: dataObject.xHighLimit
                editable: true
                implicitWidth: 170
                Layout.fillWidth: true

                ToolTip.visible: hovered
                ToolTip.text: qsTr("x from")

                textFromValue: function(value, locale) {
                    return Number(value) + " Hz"
                }

                valueFromText: function(text, locale) {
                    return parseInt(text)
                }
            }

            SelectableSpinBox {
                value: dataObject.xmax
                onValueChanged: dataObject.xmax = value
                from: dataObject.xLowLimit
                to: dataObject.xHighLimit
                editable: true
                implicitWidth: 170
                Layout.fillWidth: true

                ToolTip.visible: hovered
                ToolTip.text: qsTr("x to")

                textFromValue: function(value, locale) {
                    return Number(value) + " Hz"
                }

                valueFromText: function(text, locale) {
                    return parseInt(text)
                }
            }

            FloatSpinBox {
                id: yminFloatBox
                min: dataObject.yLowLimit
                max: dataObject.yHighLimit
                value: dataObject.ymin
                tooltiptext: qsTr("y from")
                onValueChanged: dataObject.ymin = value
                implicitWidth: 170
                Layout.fillWidth: true
                units: dataObject.yLabel
                step: 1
            }

            FloatSpinBox {
                id: ymaxFloatBox
                min: dataObject.yLowLimit
                max: dataObject.yHighLimit
                value: dataObject.ymax
                tooltiptext: qsTr("y to")
                onValueChanged: dataObject.ymax = value
                implicitWidth: 170
                Layout.fillWidth: true
                units: dataObject.yLabel
                step: 1
            }

            Connections {
                target: dataObject
                function onXminChanged() {
                    xminFloatBox.value = dataObject.xmin;
                }
                function onXmaxChanged() {
                    xmaxFloatBox.value = dataObject.xmax;
                }
                function onYminChanged() {
                    yminFloatBox.min = dataObject.yLowLimit;
                    yminFloatBox.max = dataObject.yHighLimit;
                    ymaxFloatBox.min = dataObject.yLowLimit;
                    ymaxFloatBox.max = dataObject.yHighLimit;
                    yminFloatBox.value = dataObject.ymin;
                }
                function onYmaxChanged() {
                    yminFloatBox.min = dataObject.yLowLimit;
                    yminFloatBox.max = dataObject.yHighLimit;
                    ymaxFloatBox.min = dataObject.yLowLimit;
                    ymaxFloatBox.max = dataObject.yHighLimit;
                    ymaxFloatBox.value = dataObject.ymax;
                }
                function onModeChanged() {
                    yminFloatBox.value += 0.1;
                    yminFloatBox.value -= 0.1;
                    ymaxFloatBox.value += 0.1;
                    ymaxFloatBox.value -= 0.1;
                }
            }

            CheckBox {
                id: inverse
                text: qsTr("invert")
                checked: dataObject.invert
                onCheckStateChanged: dataObject.invert = checked
                implicitWidth: 80
                ToolTip.visible: hovered
                ToolTip.text: qsTr("invert Y axis")
            }

            Button {
                font.family: "Osm"
                text: "\ue804"
                implicitWidth: 60
                onClicked: fileDialog.open();
                ToolTip.visible: hovered
                ToolTip.text: qsTr("save chart as an image")
            }
        }
        RowLayout {
            spacing: 0

            TitledCombo {
                tooltip: qsTr("Y axis mode")
                model: ["dB", "Linear"]

                currentIndex: dataObject.mode
                onCurrentIndexChanged: {
                    dataObject.mode = currentIndex;
                }
            }

            Item {
                Layout.preferredWidth: 5
            }

            TitledCombo {
                title: qsTr("ppo")
                tooltip: qsTr("points per octave")
                Layout.fillWidth: true
                model: [3, 6, 12, 24, 48]
                currentIndex: {
                    var ppo = dataObject.pointsPerOctave;
                    model.indexOf(ppo);
                }
                onCurrentIndexChanged: {
                    var ppo = model[currentIndex];
                    dataObject.pointsPerOctave = ppo;
                }
            }

            CheckBox {
                id: coherence
                text: qsTr("use coherence")
                Layout.fillWidth: true
                checked: dataObject.coherence
                onCheckStateChanged: dataObject.coherence = checked

                ToolTip.visible: hovered
                ToolTip.text: qsTr("use coherence as alpha channel")
            }

            FloatSpinBox {
                min: 0.0
                max: 1.0
                step: 0.05
                value: dataObject.coherenceThreshold
                tooltiptext: qsTr("coherence threshold")
                onValueChanged: dataObject.coherenceThreshold = value
                Layout.fillWidth: true
                visible: coherence.checked
            }

            RowLayout {
                Layout.fillWidth: true
            }

            TitledCombo {
                id: filter
                tooltip: qsTr("show only selected source")
                model: SourceModel {
                    id: filterModel
                    addNone: true
                    list: sourceList
                }
                Layout.fillWidth: true
                currentIndex: { model.indexOf(dataObject.filter) }
                textRole: "title"
                valueRole: "source"
                onCurrentIndexChanged: {
                    dataObject.filter = model.get(currentIndex);
                }
            }

            FileDialog {
                id: fileDialog
                selectExisting: false
                title: "Please choose a file's name"
                folder: (typeof shortcuts !== 'undefined' ? shortcuts.home : Filesystem.StandardFolder.Home)
                defaultSuffix: "png"
                onAccepted: {
                    dataObject.parent.grabToImage(function(result) {
                        result.saveToFile(dataObject.parent.urlForGrab(fileDialog.fileUrl));
                    });
                }
            }
        }
    }
}
