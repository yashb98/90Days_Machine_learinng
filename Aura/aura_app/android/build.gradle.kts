plugins {
    id("com.google.gms.google-services") version "4.4.0" apply false
}
allprojects {
    repositories {
        google()
        mavenCentral()
    }
}
// Reads the source
def mapsApiKey = localProperties.getProperty('GOOGLE_MAPS_API_KEY') 

// Assigns to output
manifestPlaceholders = [ geoApiKey: mapsApiKey ]

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
