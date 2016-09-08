workspace "pancakes"
  configurations { "Debug", "Release" }
  platforms "linux"

  -- Specify configuration details:
  configuration "Debug"
    targetsuffix "Dbg"
    prebuildmessage "Building Pancakes in Debug Mode."
    postbuildcommands {
      "sudo mkdir -m 0755 -p /usr/local/include/pancakes/",
      "sudo install src/includes/*.h /usr/local/include/pancakes/"
    }
  configuration "Release"
    targetsuffix ""
    prebuildmessage "Building Pancakes in Release Mode."
    postbuildcommands {
      "sudo mkdir -m 0755 -p /usr/local/include/pancakes/",
      "sudo install src/includes/*.h /usr/local/include/pancakes/"
    }



project "pancakes"
   kind "StaticLib"

   language "C++"
   targetdir "/usr/local/lib/"

   includedirs { "src/includes/" }
   files { "src/*.cpp" }

   filter "configurations:Debug"
      buildoptions { "-std=c++11", "-pedantic" }
      defines { "DEBUG" }
      symbols "On"

   filter "configurations:Release"
      buildoptions { "-std=c++11" }
      defines { "NDEBUG" }
      optimize "On"
