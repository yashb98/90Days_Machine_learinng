package com.example.aura_app

import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.util.Log

class GeospatialManager(private val context: Context) {
    private val locationManager: LocationManager? =
            context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
    private var lastLocation: Location? = null
    private val TAG = "GeospatialManager"
    private var locationCallback: ((Location) -> Unit)? = null

    private val locationListener =
            object : LocationListener {
                override fun onLocationChanged(location: Location) {
                    lastLocation = location
                    locationCallback?.invoke(location)
                    Log.d(TAG, "Location update: ${location.latitude}, ${location.longitude}")
                }

                override fun onProviderEnabled(provider: String) {
                    Log.d(TAG, "Provider enabled: $provider")
                }

                override fun onProviderDisabled(provider: String) {
                    Log.d(TAG, "Provider disabled: $provider")
                }

                @Deprecated("Deprecated in API 29")
                override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
            }

    fun startLocationUpdates(callback: (Location) -> Unit) {
        try {
            locationCallback = callback
            locationManager?.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    1000L, // minimum time between updates: 1 second
                    1f, // minimum distance: 1 meter
                    locationListener
            )
            Log.d(TAG, "Location updates started")
        } catch (e: Exception) {
            Log.e(TAG, "Error starting location updates", e)
        }
    }

    fun stopLocationUpdates() {
        try {
            locationManager?.removeUpdates(locationListener)
            Log.d(TAG, "Location updates stopped")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping location updates", e)
        }
    }

    fun getLastLocation(): Map<String, Any?> {
        return if (lastLocation != null) {
            mapOf(
                    "lat" to lastLocation!!.latitude,
                    "lng" to lastLocation!!.longitude,
                    "accuracy" to lastLocation!!.accuracy,
                    "altitude" to lastLocation!!.altitude,
                    "timestamp" to lastLocation!!.time
            )
        } else {
            mapOf("error" to "No location available yet")
        }
    }

    fun close() {
        stopLocationUpdates()
        locationCallback = null
    }
}
