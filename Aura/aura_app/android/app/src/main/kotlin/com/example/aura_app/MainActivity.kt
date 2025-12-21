package com.example.aura_app

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // 1. Register the Native AR View Factory
        // We pass the 'binaryMessenger' so the View can send data back to Flutter
        flutterEngine.platformViewsController.registry.registerViewFactory(
            "aura_ar_view",
            NativeArViewFactory(flutterEngine.dartExecutor.binaryMessenger)
        )
    }
}