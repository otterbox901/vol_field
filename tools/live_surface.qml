// The live_surface.py window: one Surface3D, refilled from Python every frame.
import QtQuick
import QtGraphs

Rectangle {
    id: root
    width: 1280
    height: 820
    color: "#101216"

    property string title: "waiting for a snapshot..."
    property string status: ""
    property real xMin: -0.6
    property real xMax: 0.6
    property real ivMin: 10
    property real ivMax: 40
    property real dMin: 1
    property real dMax: 1000

    Text {
        id: heading
        anchors { top: parent.top; left: parent.left; margins: 12 }
        text: root.title
        color: "#e6e8eb"
        font.pixelSize: 16
    }
    Text {
        anchors { top: parent.top; right: parent.right; margins: 14 }
        text: root.status
        color: "#8b919a"
        font.pixelSize: 13
        font.family: "monospace"
    }

    Surface3D {
        objectName: "graph"
        anchors { top: heading.bottom; left: parent.left; right: parent.right; bottom: parent.bottom }
        shadowQuality: Graphs3D.ShadowQuality.None
        cameraXRotation: -42
        cameraYRotation: 20
        cameraZoomLevel: 108
        aspectRatio: 1.5
        horizontalAspectRatio: 1.3
        theme: GraphsTheme {
            colorScheme: GraphsTheme.ColorScheme.Dark
            backgroundVisible: false
            labelBackgroundVisible: false
            labelTextColor: "#c4c8ce"
            labelFont.pointSize: 30
        }
        axisX: Value3DAxis {
            title: "ln K/F"
            titleVisible: true
            min: root.xMin
            max: root.xMax
            labelFormat: "%.2f"
        }
        axisY: Value3DAxis {
            title: "OTM IV %"
            titleVisible: true
            min: root.ivMin
            max: root.ivMax
            labelFormat: "%.0f"
        }
        axisZ: Value3DAxis {
            title: "days to expiry"
            titleVisible: true
            min: root.dMin
            max: root.dMax
            labelFormat: "%.0f"
            formatter: LogValue3DAxisFormatter { base: 10; autoSubGrid: true; edgeLabelsVisible: false }
        }
        Surface3DSeries {
            objectName: "series"
            drawMode: Surface3DSeries.DrawSurfaceAndWireframe
            shading: Surface3DSeries.Shading.Smooth
            wireframeColor: "#30000000"
            colorStyle: GraphsTheme.ColorStyle.RangeGradient
            baseGradient: Gradient {  // viridis
                GradientStop { position: 0.00; color: "#440154" }
                GradientStop { position: 0.25; color: "#3b528b" }
                GradientStop { position: 0.50; color: "#21918c" }
                GradientStop { position: 0.75; color: "#5ec962" }
                GradientStop { position: 1.00; color: "#fde725" }
            }
        }
    }

    // Once per rendered frame, in step with the display.
    FrameAnimation {
        running: true
        onTriggered: backend.frame()
    }
}
