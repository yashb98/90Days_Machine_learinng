package com.example.aura_app

import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.TextView
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
import io.flutter.plugin.platform.PlatformView

class NativeArView(private val context: Context, id: Int, creationParams: Map<String?, Any?>?) :
        PlatformView {

        private val arLayout: FrameLayout = FrameLayout(context)
        private val statusTextView: TextView
        private var session: Session? = null
        private var isSessionPaused = true

        init {
                // 1. Setup UI
                arLayout.setBackgroundColor(Color.BLACK)

                statusTextView =
                        TextView(context).apply {
                                text = "Initializing AR..."
                                setTextColor(Color.WHITE)
                                textSize = 16f
                                gravity = Gravity.CENTER
                                layoutParams =
                                        FrameLayout.LayoutParams(
                                                FrameLayout.LayoutParams.MATCH_PARENT,
                                                FrameLayout.LayoutParams.MATCH_PARENT
                                        )
                        }
                arLayout.addView(statusTextView)

                // 2. Initialize AR
                initArSession()
        }

        private fun initArSession() {
                try {
                        // Check if ARCore is installed
                        if (ArCoreApk.getInstance()
                                        .requestInstall(context as? android.app.Activity, true) ==
                                        ArCoreApk.InstallStatus.INSTALL_REQUESTED
                        ) {
                                return
                        }

                        // Create Session
                        session = Session(context)
                        val config = Config(session)

                        // Enable Geospatial Mode
                        config.geospatialMode = Config.GeospatialMode.ENABLED
                        config.focusMode = Config.FocusMode.AUTO
                        session?.configure(config)

                        resumeSession()
                } catch (e: Exception) {
                        statusTextView.text = "AR Init Failed: ${e.message}"
                        e.printStackTrace()
                }
        }

        private fun resumeSession() {
                try {
                        if (session != null && isSessionPaused) {
                                session?.resume()
                                isSessionPaused = false
                                statusTextView.text = "AR Session Running\n(Geospatial Mode)"
                        }
                } catch (e: CameraNotAvailableException) {
                        statusTextView.text = "Camera Unavailable"
                } catch (e: Exception) {
                        statusTextView.text = "Resume Failed: ${e.message}"
                }
        }

        override fun getView(): View {
                return arLayout
        }

        override fun dispose() {
                session?.pause()
                session?.close()
                session = null
        }
}
