workspace "hello_pancakes"
  configurations { "Debug", "Release" }

project "hello_pancakes"
   kind "ConsoleApp"

   language "C++"
   targetdir "bin/"


   files { "src/*.cpp", "src/*.h" }

   filter "configurations:Debug"
      buildoptions { "-std=c++11", "-pedantic", "-libpancakes" }
      defines { "DEBUG" }
      symbols "On"

   filter "configurations:Release"
      buildoptions { "-std=c++11", "-libpancakes" }
      defines { "NDEBUG" }
      optimize "On"
