package com.example.aura_app

import android.content.Context
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.TextView
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.platform.PlatformView

class NativeArView(
    private val context: Context, 
    id: Int, 
    creationParams: Map<String?, Any?>?, 
    messenger: BinaryMessenger
) : PlatformView {

    private val arLayout: FrameLayout = FrameLayout(context)
    private val statusTextView: TextView
    private var session: Session? = null
    
    // Channel to talk to Flutter
    private val channel = MethodChannel(messenger, "aura/geospatial")
    
    // Loop handlers
    private val handler = Handler(Looper.getMainLooper())
    private var isTracking = false

    init {
        // 1. Setup UI (Simple Placeholder for now)
        arLayout.setBackgroundColor(Color.BLACK)
        statusTextView = TextView(context).apply {
            text = "Initializing Aura AR..."
            setTextColor(Color.WHITE)
            textSize = 16f
            gravity = Gravity.CENTER
        }
        arLayout.addView(statusTextView)

        // 2. Handle Commands from Flutter (start/stop)
        channel.setMethodCallHandler { call, result ->
            when (call.method) {
                "startArSession" -> {
                    startTracking()
                    result.success(null)
                }
                "stopArSession" -> {
                    stopTracking()
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        }

        // 3. Auto-start AR initialization
        initArSession()
    }

    private fun initArSession() {
        try {
            if (ArCoreApk.getInstance().requestInstall(context as? android.app.Activity, true) ==
                ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                return
            }

            session = Session(context)
            val config = Config(session)
            
            // CRITICAL: Enable Geospatial Mode
            config.geospatialMode = Config.GeospatialMode.ENABLED
            config.focusMode = Config.FocusMode.AUTO
            session?.configure(config)
            
            resumeSession()
        } catch (e: Exception) {
            statusTextView.text = "AR Error: ${e.message}"
        }
    }

    private fun resumeSession() {
        try {
            session?.resume()
            startTracking() // Start the data loop
        } catch (e: CameraNotAvailableException) {
            statusTextView.text = "Camera Unavailable"
        } catch (e: Exception) {
            statusTextView.text = "Resume Failed: ${e.message}"
        }
    }

    // --- DATA LOOP ---
    private val updateRunnable = object : Runnable {
        override fun run() {
            if (!isTracking || session == null) return

            try {
                // 1. Update ARCore Frame
                // Note: Without a GLSurfaceView, this might not render the camera feed,
                // but we need it to process the sensors.
                session?.update()

                val earth = session?.earth
                
                // 2. Check if Earth Localization is ready
                if (earth?.trackingState == TrackingState.TRACKING && 
                    earth.cameraGeospatialPose != null) {
                    
                    val pose = earth.cameraGeospatialPose
                    
                    // 3. Prepare Data for Flutter
                    val data = mapOf(
                        "lat" to pose.latitude,
                        "lng" to pose.longitude,
                        "heading" to pose.heading,
                        "altitude" to pose.altitude,
                        "accuracy" to pose.horizontalAccuracy
                    )

                    // 4. Send to Flutter
                    channel.invokeMethod("onLocationUpdate", data)
                    
                    // Update Debug UI
                    statusTextView.text = "AR TRACKING ACTIVE\nLat: ${pose.latitude}\nLng: ${pose.longitude}"
                } else {
                    statusTextView.text = "Waiting for VPS Localization..."
                }
            } catch (e: Exception) {
                // Ignore frame errors
            }

            // Loop every 500ms
            handler.postDelayed(this, 500)
        }
    }

    private fun startTracking() {
        if (!isTracking) {
            isTracking = true
            handler.post(updateRunnable)
        }
    }

    private fun stopTracking() {
        isTracking = false
        session?.pause()
        handler.removeCallbacks(updateRunnable)
    }

    override fun getView(): View {
        return arLayout
    }

    override fun dispose() {
        stopTracking()
        session?.close()
        session = null
    }
}