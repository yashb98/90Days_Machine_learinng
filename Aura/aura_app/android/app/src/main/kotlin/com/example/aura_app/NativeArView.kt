package com.example.aura_app

import android.content.Context
import android.graphics.Color
import android.view.View
import android.widget.FrameLayout
import android.widget.TextView
import com.google.ar.core.Session
import io.flutter.plugin.platform.PlatformView

class NativeArView(context: Context, id: Int, creationParams: Map<String?, Any?>?) : PlatformView {

        private val arLayout: FrameLayout = FrameLayout(context)
        private val statusTextView: TextView

        // ARCore Session
        private var session: Session? = null

        init {
                // 1. Setup Basic UI (Placeholder for now)
                arLayout.setBackgroundColor(Color.BLACK)

                statusTextView = TextView(context)
                statusTextView.text = "Initializing AR Core..."
                statusTextView.setTextColor(Color.WHITE)
                statusTextView.textSize = 20f
                statusTextView.textAlignment = View.TEXT_ALIGNMENT_CENTER

                arLayout.addView(statusTextView)

                // 2. Try Initialize AR (Basic check)
                try {
                        // We will initialize the real AR session in the onResume/Lifecycle methods
                        // For now, just checking if we can reference the class ensures the library
                        // is linked.
                        statusTextView.text = "AR View Ready (Waiting for Session)"
                } catch (e: Exception) {
                        statusTextView.text = "AR Error: ${e.message}"
                }
        }

        override fun getView(): View {
                return arLayout
        }

        override fun dispose() {
                session?.close()
                session = null
        }
}
