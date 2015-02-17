package jmetal.metaheuristics.ibea;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import jmetal.core.SolutionSet;
import jmetal.opencl.*;

import org.jocl.*;

import static org.jocl.CL.*;
public class IBEAOpenCL {
	
	private int population_size;
	private OpenCLEnvironment openCLInst;
	
	private double indicatorValues[]; 
	private Pointer p_indicatorValues;
	private cl_mem mem_indicatorValues;
	
	private double fitnessValues[];
	private Pointer p_fitnessValues;
	private cl_mem mem_fitnessValues;
	
	private double objectiveValues[];
	private Pointer p_objectiveValues;
	private cl_mem mem_objectiveValues;
	

	private cl_mem mem_minimum;

	private cl_mem mem_maximum;
	
	private cl_program program;
	private cl_kernel fitness_kernel;
	private cl_kernel indicator_kernel;
	
	public IBEAOpenCL(int pop_size){
		
		
		population_size = pop_size;
		openCLInst = OpenCLEnvironment.getInstance();
		cl_context context = openCLInst.getContext();
		// Build kernel code
		// Create the program from the source code
       // program = clCreateProgramWithSource(context,
       //     1, new String[]{ programSource }, null, null);
		
        program = clCreateProgramWithSource(context,
                1, new String[]{ programSource }, null, null);
        // Build the program
        clBuildProgram(program, 0, null, null, null, null);
        
        // Create the kernel
        fitness_kernel = clCreateKernel(program, "fitness_kernel", null);
        indicator_kernel =  clCreateKernel(program, "indicator_kernel", null);
		// Prepare all buffers
		
		indicatorValues = new double[(pop_size * 2) * (pop_size * 2)];
		p_indicatorValues = Pointer.to(indicatorValues);
		
		mem_indicatorValues = clCreateBuffer(openCLInst.getContext(), 
                CL_MEM_READ_WRITE, 
                Sizeof.cl_double * (pop_size * 2) * (pop_size * 2) , null, null);
		
		fitnessValues = new double[(pop_size * 2)];
		p_fitnessValues = Pointer.to(fitnessValues);
		mem_fitnessValues = clCreateBuffer(openCLInst.getContext(), 
                CL_MEM_READ_WRITE, 
                Sizeof.cl_double * (pop_size * 2) , null, null);
		
		int max_objective_number = 2; // This is hardcore.
		objectiveValues = new double[(pop_size * 2) * max_objective_number];
		p_objectiveValues = Pointer.to(objectiveValues);
		mem_objectiveValues = clCreateBuffer(openCLInst.getContext(), 
                CL_MEM_READ_ONLY, 
                Sizeof.cl_double * (pop_size * 2) * max_objective_number , null, null);
		
		
		mem_maximum = clCreateBuffer(openCLInst.getContext(), 
                CL_MEM_READ_ONLY, 
                Sizeof.cl_double * max_objective_number , null, null);
		
		mem_minimum = clCreateBuffer(openCLInst.getContext(), 
                CL_MEM_READ_ONLY, 
                Sizeof.cl_double * max_objective_number , null, null);
		
		
	}
	public double calcFitness(List<List<Double>>  lIndicatorValues, SolutionSet solutionSet, 
				double[] maximumValues, double[] minimumValues){
		
		int solution_size = solutionSet.size();
	
		int objective_num = solutionSet.get(0).getNumberOfObjectives();
		
		for (int i = 0; i < solution_size; i++){
			for (int j = 0; j < objective_num; j++)
			{
				objectiveValues[j * (population_size * 2) + i] = solutionSet.get(i).getObjective(j);
			}
		}
		
		
		clEnqueueWriteBuffer(openCLInst.getCommandQueue(), mem_objectiveValues, CL_TRUE, 0,
				(population_size * 2) * objective_num * Sizeof.cl_double, p_objectiveValues, 0, null, null);
		
		clEnqueueWriteBuffer(openCLInst.getCommandQueue(), mem_maximum, CL_TRUE, 0,
				objective_num * Sizeof.cl_double, Pointer.to(maximumValues), 0, null, null);
		
		clEnqueueWriteBuffer(openCLInst.getCommandQueue(), mem_minimum, CL_TRUE, 0,
				objective_num * Sizeof.cl_double, Pointer.to(minimumValues), 0, null, null);
		///////////HyperVolumeIndicator///////////
        // Set launch configurations

        
        long globalWorkSize0[] = new long[2];
        globalWorkSize0[0] = solution_size;
        globalWorkSize0[1] = solution_size;
        //Set kernel parameter
        clSetKernelArg(indicator_kernel, 0, 
                Sizeof.cl_mem, Pointer.to(mem_objectiveValues)); 
        clSetKernelArg(indicator_kernel, 1, 
                Sizeof.cl_mem, Pointer.to(mem_indicatorValues)); 
        clSetKernelArg(indicator_kernel, 2, 
                Sizeof.cl_mem, Pointer.to(mem_maximum));
        clSetKernelArg(indicator_kernel, 3, 
                Sizeof.cl_mem, Pointer.to(mem_minimum));
        clSetKernelArg(indicator_kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{objective_num}));
        clSetKernelArg(indicator_kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{solution_size}));
        clSetKernelArg(indicator_kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{population_size}));
        
        
        clEnqueueNDRangeKernel(openCLInst.getCommandQueue(), indicator_kernel, 2, null,
                globalWorkSize0, null , 0, null, null);
        
        // Copy Indicator to host, because removeWorst still use it.
        clEnqueueReadBuffer(openCLInst.getCommandQueue(), mem_indicatorValues, CL_TRUE, 0,
                solution_size * solution_size  * Sizeof.cl_double, p_indicatorValues, 0, null, null);
        double maxIndicatorValue =  - Double.MAX_VALUE;
       // lIndicatorValues = new ArrayList<List<Double>>();
        for (int i = 0; i < solution_size; i++) {
          
            List<Double> aux = new ArrayList<Double>();
            for (int j = 0; j < solution_size; j++) {
            	double value = indicatorValues[j * solution_size + i];
                if (Math.abs(value) > maxIndicatorValue)
                	maxIndicatorValue = Math.abs(value);
                aux.add(value);
            }
            lIndicatorValues.add(aux);
        }
   

        //////////fitness//////////////
        // Set launch configurations
        long localWorkSize[] = new long[1];
        localWorkSize[0] = 128;
        
        long globalWorkSize[] = new long[1];
        globalWorkSize[0] = solution_size * localWorkSize[0];
        //Set kernel parameter
      
        clSetKernelArg(fitness_kernel, 0, 
                Sizeof.cl_mem, Pointer.to(mem_indicatorValues));        
        clSetKernelArg(fitness_kernel, 1, 
                        Sizeof.cl_mem, Pointer.to(mem_fitnessValues));
        clSetKernelArg(fitness_kernel, 2, Sizeof.cl_double, Pointer.to(new double[]{maxIndicatorValue}));
        clSetKernelArg(fitness_kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{solution_size}));
        clSetKernelArg(fitness_kernel, 4, Sizeof.cl_double * localWorkSize[0], null);
        
        //Launch kernel
        
        clEnqueueNDRangeKernel(openCLInst.getCommandQueue(), fitness_kernel, 1, null,
                globalWorkSize, localWorkSize , 0, null, null);
        
      //Transfer result to host
        clEnqueueReadBuffer(openCLInst.getCommandQueue(), mem_fitnessValues, CL_TRUE, 0,
                solution_size  * Sizeof.cl_double, p_fitnessValues, 0, null, null);
        
      for (int pos =0; pos < solution_size ; pos++) {
    	  double fitness = fitnessValues[pos];
             
          solutionSet.get(pos).setFitness(fitness);
      //    System.out.println(pos +"="+ fitness);
      }
      
      return maxIndicatorValue;
        	
	}
	public void release(){
		clReleaseMemObject(mem_maximum);
		clReleaseMemObject(mem_minimum);
		clReleaseMemObject(mem_objectiveValues);
		clReleaseMemObject(mem_indicatorValues);
		clReleaseMemObject(mem_fitnessValues);
        clReleaseKernel(fitness_kernel);
        clReleaseKernel(indicator_kernel);
        clReleaseProgram(program);
        OpenCLEnvironment.getInstance().free();
	}
	
    private static String programSource =
    		"    		#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n"+
    				"            __kernel void "+
    				"            fitness_kernel(__global double *indicatorValues,"+
    				"                         __global double * fitnessValues,"+
    				"                         double maxIndicatorValue, int s_size, __local double * shared)\n"+
    				"            {\n"+
    				"                int pos = get_group_id(0);\n"+
    				"                int i = get_local_id(0);\n"+
    				"                shared[i] = 0;\n"+
    				"            	 barrier(CLK_LOCAL_MEM_FENCE);\n"+
    				"            	 while (i < s_size){\n"+
    				"                   if (i!=pos)\n"+
    				"            			shared[get_local_id(0)] += "+
    				"            				exp((-1 * indicatorValues[pos * s_size + i]/maxIndicatorValue) / 0.05);\n"+
    				"                   i += get_local_size(0);\n"+
    				"            	 }\n"+
    				"                barrier(CLK_LOCAL_MEM_FENCE);\n"+
    				"                i = get_local_id(0); \n"+
    				"                for (int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2){\n"+
    				"                	if (i < offset){\n"+
    				"                     shared[i] += shared[i + offset];\n"+
    				"                   }\n"+
    				"                   barrier(CLK_LOCAL_MEM_FENCE);\n"+
    				"                }\n"+
    				"                if (i == 0) \n"+
    				"                   fitnessValues[pos] = shared[0];\n"+
    				"            }\n"+
    				"            __kernel void  "+
    				"            indicator_kernel(__global double * objValues, __global double * indicatorValues, "+
    				"                              __global double * maximum, __global double * minimum, "+
    				"                               int obj_num, int s_size, int pop_size){\n"+
    				"                int idx = get_global_id(0);\n"+
    				"                int jdx = get_global_id(1);\n"+
    				"                if (idx < s_size && jdx < s_size){\n"+
    				"                    int flag; \n"+
    				"                    double value1, value2;\n"+
    				"                    int dominate1 = 0;\n"+
    				"                    int dominate2 = 0;\n"+                  
    				"   				    for (int i = 0; i < obj_num; i++) {\n"+
    				"	                       value1 = objValues[i * (pop_size * 2) + idx];\n"+
    				"	                       value2 = objValues[i * (pop_size * 2) + jdx];\n"+
    				"	      				   if (value1 < value2) {\n"+
    				"	        					flag = -1;\n"+
    				"					      } else if (value1 > value2) {\n"+
    				"					        	flag = 1;\n"+
    				"					      } else {\n"+
    				"					       	    flag = 0;\n"+
    				"					      }\n"+
    				"					      "+
    				"					      if (flag == -1) {\n"+
    				"					        dominate1 = 1;\n"+
    				"					      }\n"+
    				"					      \n"+
    				"					      if (flag == 1) {\n"+
    				"					        dominate2 = 1; \n"+          
    				"					      }\n"+
    				"  				 	}        \n"+
    				"   					if (dominate1 == dominate2)\n"+ 
    				"   					 	flag = 0;\n"+
    				"   					else if ( dominate1 == 1)\n"+
    				"   					 	flag = -1;\n"+
    				"   					else flag = 1; \n"+
    				"   					"+
    				"   					int tidx = idx;\n"+
    				"   					int tjdx = jdx;\n"+
    				"   					if (flag != -1){\n"+
    				"   						tidx = jdx;\n"+
    				"   						tjdx = idx;\n"+
    				"   					}  					 \n"+  					
    				"   					double volume = 0;\n"+
    				"   					double max, a, b;\n"+
    				"   					double r = 2.0 * (maximum[obj_num - 1] - minimum[obj_num - 1]);\n"+
    				"   		            max = minimum[obj_num - 1] + r;\n"+                 
    				"					a = objValues[(obj_num-1) * (pop_size * 2) + tidx];\n"+
    				"					b = objValues[(obj_num-1) * (pop_size * 2) + tjdx];\n"+
    				"					                \n"+
    				"					if (obj_num == 1) {\n"+
    				"						if (a < b)\n"+
    				"							volume = (b - a) / r;\n"+
    				"						else\n"+
    				"							volume = 0;\n"+
    				"					} else {\n"+
    				"							if (a < b) {\n"+
    				"								double r1 = 2.0 * (maximum[0] - minimum[0]);\n"+
    				"								\n"+
    				"								double max1 = minimum[0] + r1;\n"+
    				"								double a1 =  objValues[tidx];\n"+
    				"								double b1 = max1;\n"+
    				"								if (a1 < b1)\n"+
    				"									volume = (b1 - a1) / r1;\n"+
    				"								else\n"+
    				"									volume = 0;\n"+
    				"								volume *= (b - a) / r;\n"+
    				"				\n"+
    				"								b1 = objValues[tjdx];\n"+
    				"								double volume1 = 0;\n"+
    				"								if (a1 < b1)\n"+
    				"									volume1 = (b1 - a1) / r1;\n"+
    				"								else\n"+
    				"									volume1 = 0;\n"+
    				"								volume += volume1 * (max - b) / r;\n"+
    				"						\n"+
    				"							} else {\n"+
    									
    				"								double r1 = 2.0 * (maximum[0] - minimum[0]);\n"+
    				"								double a1 = objValues[tidx];\n"+
    				"								double b1 = objValues[tjdx];		\n"+		                
    				"								if (a1 < b1)\n"+
    				"										volume = (b1 - a1) / r1;\n"+
    				"								else\n"+
    				"										volume = 0;\n"+
    				"								volume *= (max - b) / r;\n"+
    				"							}\n"+
    				"					}				\n"+
    				"					if (flag == -1)\n"+
    				"					   indicatorValues[jdx * s_size + idx] =- volume;\n"+
    				"					else\n"+
    				"					   indicatorValues[jdx * s_size + idx] =  volume;	\n"+		   					 	         
    				"    			}   \n"+
    				"            }\n"
            ;
        
   
}
