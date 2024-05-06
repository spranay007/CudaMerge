# CudaMerge.cu

The goal of this project is to implement merge sort algorithm using shared memory in CUDA to make the sorting process faster

## Platform
Project created on Windows X64

## IDE Used
Visual Studio 2022

## Build the code 
-> Extract the zip folder on a Windows machine 
-> Build using Visual Studio 2022 [use the 2022 version]
-> Now open the terminal and traverse to the Debug directory.

## Usage
Sample command to create 
To Sort: randomly generated 100000 numbers 
******************************************
```bash
cd .\x64\
cd .\Debug\

# Now in the Debug directory
.\CudaMerge.exe 100000
```
******************************************
## NOTE
Please make sure that the project settings are also imported, if not then follow this
******************************************
--> Open Solution Explorer 
--> Navigate to projectOne properties page 
--> Under VC++ Directories 
--> Find External Included Directories 
--> Add the Project_three\Common folder to the list(Already present in the Zipped folder) 
--> Necessary to include 
    #include <helper_functions.h>
    #include <helper_cuda.h>
