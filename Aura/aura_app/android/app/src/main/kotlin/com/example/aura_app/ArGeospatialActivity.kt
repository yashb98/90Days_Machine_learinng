// In your session configuration
val config = Config(session)

config.geospatialMode = Config.GeospatialMode.ENABLED

session.configure(config)

val earth = session.earth

if (earth?.trackingState == TrackingState.TRACKING) {
    val cameraGeospatialPose = earth.cameraGeospatialPose
    // Now you have highly accurate Lat/Lng/Heading
}

// Place an anchor at the bus stop coordinates
val busStopAnchor =
        earth.createAnchor(
                busStopLatitude,
                busStopLongitude,
                earth.cameraGeospatialPose.altitude, // Anchor at device altitude initially
                0f,
                0f,
                0f,
                1f // Rotation quaternion
        )

// Attach a 3D marker (e.g., a floating "BUS" icon) to this anchor
