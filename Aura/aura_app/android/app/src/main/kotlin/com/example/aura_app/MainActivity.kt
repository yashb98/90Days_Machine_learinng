package com.example.aura_app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private val CHANNEL = "aura/geospatial"

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // 1. Register the Native AR View
        flutterEngine.platformViewsController.registry.registerViewFactory(
                "aura_ar_view",
                NativeArViewFactory()
        )

        // 2. Register the Method Channel (For logic commands)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler {
                call,
                result ->
            if (call.method == "startArSession") {
                // We will implement the actual session start logic later
                println("🟢 Native received: startArSession")
                result.success("AR Session Started")
            } else if (call.method == "stopArSession") {
                println("🔴 Native received: stopArSession")
                result.success("AR Session Stopped")
            } else {
                result.notImplemented()
            }
        }
    }
}
