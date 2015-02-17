package jmetal.opencl;

import static org.jocl.CL.*;

import java.util.logging.Logger;

import org.jocl.*;

public class OpenCLEnvironment {
	private static OpenCLEnvironment instance = null;
	protected OpenCLEnvironment(){
		
	}
	
	public static OpenCLEnvironment getInstance(){
		if(instance == null) {
	         instance = new OpenCLEnvironment();
	         instance.setUp();
	      }
	      return instance;
	}
	
	
	private cl_command_queue commandQueue;
	private cl_context context ;
	
	public cl_command_queue getCommandQueue(){
		return commandQueue;
	}
	public cl_context getContext(){
		return context;
	}
	
	private void setUp(){
        // The platform, device type and device number
        // that will be used
        int platformIndex = -1;
        final long deviceType = CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        
        // Set priority to choose the platform : AMD > NVIDIA > Intel
        String platformPriorityName [] = { "amd", "nvidia", "intel" };
        
        int highestPriority = Integer.MAX_VALUE;
        
        int platformPriorities[] = new int[numPlatforms];
        
        // Explore all platforms
		for (int i = 0; i < numPlatforms; i++) {

			// Obtain the length of the string that will be queried
			long size[] = new long[1];
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, null, size);

			// Create a buffer of the appropriate size and fill it with the info
			byte buffer[] = new byte[(int) size[0]];
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, buffer.length,
					Pointer.to(buffer), null);

			// Create a string from the buffer (excluding the trailing \0 byte)
			String platformName = new String(buffer, 0, buffer.length - 1);
			platformName = platformName.toLowerCase();
			
			// Assign true value for each detected platform.
			for (int p = 0 ; p < platformPriorityName.length ; p++)
				if (platformName.contains(platformPriorityName[p])){
					platformPriorities[i] = p;
					
					if (highestPriority > p) highestPriority = p;
					break;
				}
		
		}
        // Select platform with highest priority
		for (int i = 0 ; i < numPlatforms; i++){
			if (platformPriorities[i] == highestPriority){
				//platformIndex = i;
				platformIndex = 1;
				break;
			}
		}
		
        cl_platform_id platform = platforms[platformIndex];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        
        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];
        
        // Obtain a device ID 
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        String deviceName = getString(device, CL_DEVICE_NAME);
        System.out.println(deviceName);
        // Create a context for the selected device
        context = clCreateContext(
            contextProperties, 1, new cl_device_id[]{device}, 
            null, null, null);
        
        // Create a command-queue for the selected device
        commandQueue = 
            clCreateCommandQueue(context, device, 0, null);
	}
	
	private static String getString(cl_device_id device, int paramName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }
	
	public  void free()
	{

        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
	}

}
