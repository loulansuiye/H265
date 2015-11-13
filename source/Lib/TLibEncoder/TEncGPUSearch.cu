#include "TEncGPUSearch.h"
#include "TEncMEs.h"
#include <pthread.h>
#include <math.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>

//From TComRom.h 
#define     MAX_CU_DEPTH             6                          // log2(CTUSize)
#define     MAX_CU_SIZE             (1<<(MAX_CU_DEPTH))  
#define 	FILTER_BLOCK_STRIDE 80
#define 	HALF_FITLER_SIZE  4

// This macro is for checking return codes
// from a cuda function and print the error
// message to the stderr. The key is cudaGetErrorString
// If it is not used on every cuda function, it
// could report error message due to previous failed calls.

#define CUDART_CHECK( fn ) do { \
		status = (fn); \
		if ( cudaSuccess != (status) ) { \
			fprintf( stderr, "CUDA Runtime Failure (line %d of file %s):\n\t" \
			"%s returned 0x%x (%s)\n", \
			__LINE__, __FILE__, #fn, status, cudaGetErrorString(status) ); \
			exit(1); \
		} \
	} while (0); 
 

//Allocate Templates
//Initialize all templates 
//Host
template void GPU::MemOpt::AllocHostMem<UInt*>(UInt*&, size_t); //Page-locked

//Device
template void GPU::MemOpt::AllocDeviceMem<GpuMeDataAccess*>(GpuMeDataAccess*&, size_t);
template void GPU::MemOpt::AllocDeviceMem<Pel*>(Pel*&, size_t);
template void GPU::MemOpt::AllocDeviceMem<UInt*>(UInt*&, size_t);


//TransferToDevice Templates
template void GPU::MemOpt::TransferToDevice<Pel*>(Pel*&, Pel*&, size_t);
template void GPU::MemOpt::TransferToDevice<UInt*>(UInt*&, UInt*&, size_t); //For debugging purpose
template void GPU::MemOpt::TransferToDevice<GpuMeDataAccess*>(GpuMeDataAccess*&, GpuMeDataAccess*&, size_t);

//TransferToHost Templates
template void GPU::MemOpt::TransferToHost<Pel*>(Pel*&, Pel*&, size_t);
template void GPU::MemOpt::TransferToHost<UInt*>(UInt*&, UInt*&, size_t);


//Free templates
//Host
template void GPU::MemOpt::FreeHostMem<UInt>(UInt *);

//Device
template void GPU::MemOpt::FreeDeviceMem<Pel>(Pel *);
template void GPU::MemOpt::FreeDeviceMem<GpuMeDataAccess>(GpuMeDataAccess *);
template void GPU::MemOpt::FreeDeviceMem<UInt>(UInt *); 


// Multithreading function goes in here
namespace CPU
{

}

// GPU functions goes in here
namespace GPU
{
	namespace MemOpt
	{

		Int 
		ChangeDevice(Int iDevice) {
			Int deviceCount = 1;
			Int setDevice;
			cudaError_t status;

			setDevice = iDevice % deviceCount;
    		CUDART_CHECK( cudaGetDeviceCount(&deviceCount) );
			CUDART_CHECK( cudaSetDevice(setDevice) );
			return setDevice;
		}

		template<class T>
		void
		AllocDeviceMem(T& src, size_t len)
		{
			cudaError_t status;
			CUDART_CHECK( cudaMalloc((void**)&src, len));
		}

		template<class T>
		void
		AllocHostMem(T& src, size_t len)
		{
			cudaError_t status;
			CUDART_CHECK( cudaMallocHost((void**)&src, len));
		}


		template<class T>
		void
		TransferToDevice(T& src, T& targ, size_t len)
		{
			cudaError_t status;
			CUDART_CHECK( cudaMemcpy(targ, src, len, cudaMemcpyHostToDevice));
		}

		template<class T>
		void
		TransferToHost(T& src, T& targ, size_t len)
		{
			cudaError_t status;
			CUDART_CHECK( cudaMemcpy(targ, src, len, cudaMemcpyDeviceToHost));
		}

		template<class T>
		void
		FreeDeviceMem(T *src)
		{
			cudaFree(src);
		}

		template<class T>
		void
		FreeHostMem(T *src)
		{
			cudaFreeHost(src);
		}
		
	}

	namespace InterpolationKernel
	{
		/*
		__constant__ int acMvRefineHQ[9][2] =
		{
			{ 0, 0},
			{ 0,-1},
			{ 0, 1},
			{-1, 0},
			{ 1, 0},
			{-1,-1},
			{ 1,-1},
			{-1, 1},
			{ 1, 1}
		};
		__constant__ int lumaFilter[4][8] =
		{
		  {  0, 0,   0, 64,  0,   0, 0,  0 },
		  { -1, 4, -10, 58, 17,  -5, 1,  0 }, //m_lumaFilter[1] quarter sample
		  { -1, 4, -11, 40, 40, -11, 4, -1 }, //m_lumaFilter[2] half sample
		  {  0, 1,  -5, 17, 58, -10, 4, -1 }
		};

		__global__ void gpuGpuHalfSampleMEKernel(Int iPUWidth, 
												 Int iPUHeight,
												 Int extWidth,
												 Int extHeight,
 												 Pel* srcPtr, //Pointed to the integer MV
												 Pel* m_pYOrgPU,
												 UInt m_uiCost,
												 Int  m_iCostScale,
												 Int  m_iMvPredx,
												 Int  m_iMvPredy,
												 Int iStride,
												 Pel *filteredBlock)
		{
			__shared__ Pel tmp[2][80*80];

			Pel* dst, *src;
			Int x,y;

			//Filter Integer Pixels
			for(x = 0; x < iPUHeight + 8; x++)
			{
				tmp[0][x*80 + threadIdx.x] = (srcPtr[x*iStride + threadIdx.x] << 6) -  8192;
			}

			//Filter "b"s
			//b(0,0) = -A(-3,0) + 4*A(-2,0) - 11*A(-1,0) + 40*A(0,0) + 40*A(1,0) - 11*A(2,0) + 4*A(3,0) -A(4,0)  
			src = srcPtr - 3; 
			for(x = 0; x < iPUHeight + 8; x++)
			{
				y  = src[x*iStride + threadIdx.x + 0]*lumaFilter[2][0];
				y += src[x*iStride + threadIdx.x + 1]*lumaFilter[2][1];
				y += src[x*iStride + threadIdx.x + 2]*lumaFilter[2][2];
				y += src[x*iStride + threadIdx.x + 3]*lumaFilter[2][3];
				y += src[x*iStride + threadIdx.x + 4]*lumaFilter[2][4];
				y += src[x*iStride + threadIdx.x + 5]*lumaFilter[2][5];
				y += src[x*iStride + threadIdx.x + 6]*lumaFilter[2][6];
				y += src[x*iStride + threadIdx.x + 7]*lumaFilter[2][7];
				
				tmp[1][x*80 + threadIdx.x] = y - 8192;
			}

			//Copy integer into output
			if(blockIdx.x==0)
			{
				dst = filteredBlock;
				src = tmp[0] + HALF_FITLER_SIZE * FILTER_BLOCK_STRIDE + 1;

				for(x = 0; x < iPUHeight; x++)
				{
					y = (tmp[0][x*80 + threadIdx.x] + 8192) << 6;
					y = (y < 0) ? 0: y;
					y = (y > 255) ? 255 : y;
					dst[threadIdx.x + x*FILTER_BLOCK_STRIDE] = y;
				}
			}

			//Copy "b"s into output
			if(blockIdx.x==1)
			{
				dst = filteredBlock + FILTER_BLOCK_STRIDE*FILTER_BLOCK_STRIDE;
				src = tmp[1] + HALF_FITLER_SIZE * FILTER_BLOCK_STRIDE;

				for(x = 0; x< iPUHeight; x++)
				{
					y = (src[threadIdx.x] + 8192) >> 6;
					//Check bounds
					y = (y < 0) ? 0: y;
					y = (y > 255) ? 255 : y;
					dst[threadIdx.x + x*x*FILTER_BLOCK_STRIDE] = y;

				}
			}

			//Filter "h"s
			//h(0,0) = -A(0,-3) + 4*A(0,-2) - 11*A(0,-1) + 40*A(0,0) + 40*A(0,1) - 11*A(0,2) + 4*A(0,3) -A(0,4) 
			if(blockIdx.x==2)
			{
				//This will result in 2-way bank conflict. 
				dst = filteredBlock + 2*FILTER_BLOCK_STRIDE*FILTER_BLOCK_STRIDE;
				src = tmp[0] + (HALF_FITLER_SIZE - 1) * FILTER_BLOCK_STRIDE + 1 - 3 * FILTER_BLOCK_STRIDE;

				for(x = 0; x < iPUHeight + 1; x++)
				{
					y  = src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 0*FILTER_BLOCK_STRIDE]*lumaFilter[2][0];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 1*FILTER_BLOCK_STRIDE]*lumaFilter[2][1];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 2*FILTER_BLOCK_STRIDE]*lumaFilter[2][2];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 3*FILTER_BLOCK_STRIDE]*lumaFilter[2][3];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 4*FILTER_BLOCK_STRIDE]*lumaFilter[2][4];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 5*FILTER_BLOCK_STRIDE]*lumaFilter[2][5];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 6*FILTER_BLOCK_STRIDE]*lumaFilter[2][6];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 7*FILTER_BLOCK_STRIDE]*lumaFilter[2][7];
					y = (y + 526336) >> 12;
					
					//Check bounds
					y = (y < 0) ? 0: y;
					y = (y > 255) ? 255 : y;
					dst[x*FILTER_BLOCK_STRIDE + threadIdx.x] = y;

				}

			}

			//Filter "j"s (uses "b"s)
			if(blockIdx.x==3)
			{
				dst = filteredBlock + 3*FILTER_BLOCK_STRIDE*FILTER_BLOCK_STRIDE;
				src = tmp[1] + (HALF_FITLER_SIZE - 1) * FILTER_BLOCK_STRIDE - 3 * FILTER_BLOCK_STRIDE;

				for(x = 0; x < iPUHeight + 1; x++)
				{
					y  = src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 0*FILTER_BLOCK_STRIDE]*lumaFilter[2][0];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 1*FILTER_BLOCK_STRIDE]*lumaFilter[2][1];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 2*FILTER_BLOCK_STRIDE]*lumaFilter[2][2];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 3*FILTER_BLOCK_STRIDE]*lumaFilter[2][3];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 4*FILTER_BLOCK_STRIDE]*lumaFilter[2][4];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 5*FILTER_BLOCK_STRIDE]*lumaFilter[2][5];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 6*FILTER_BLOCK_STRIDE]*lumaFilter[2][6];
					y += src[x*FILTER_BLOCK_STRIDE + threadIdx.x + 7*FILTER_BLOCK_STRIDE]*lumaFilter[2][7];
					
					dst[x*FILTER_BLOCK_STRIDE + threadIdx.x] = (y + 526336) >> 12;

				}


			}

		}

		void gpuGpuHalfSampleME(GpuMeDataAccess* pHostGMDA, Int iStride, Int iMvX, Int iMvY)
		{
			
			cudaError_t status;
			Pel* piROIOrigin;
			Pel* gpufilteredBlock;
			Pel* cpufilteredBlock;
			Pel* srcPtr;
			Int  iOffset    = iMvX  + iMvY  * iStride;
  			Int extWidth  = MAX_CU_SIZE + 16;
    		Int extHeight = MAX_CU_SIZE + 1;
    		Int halfFilterSize = 4;

			if(pHostGMDA->m_iRefListId==0)
				piROIOrigin = pHostGMDA->m_cYRefList0[pHostGMDA->m_iRefId] + pHostGMDA->getLOrgOffset() + pHostGMDA->m_iPUOffset;
			else
				piROIOrigin = pHostGMDA->m_cYRefList1[pHostGMDA->m_iRefId] + pHostGMDA->getLOrgOffset() + pHostGMDA->m_iPUOffset;

 			piROIOrigin += iOffset; //piROIOrigin points to the integer MV location
 			srcPtr = piROIOrigin - halfFilterSize*iStride - 1;
 			 
 			cpufilteredBlock = new Pel[4*FILTER_BLOCK_STRIDE*FILTER_BLOCK_STRIDE]; //4 location for the half sample ME
			GPU::MemOpt::AllocDeviceMem<Pel*>(gpufilteredBlock, 4*FILTER_BLOCK_STRIDE*FILTER_BLOCK_STRIDE);


			int iNumThreads =  pHostGMDA->m_iPUWidth;
			int iNumBlocks  = 4; //Number of search locations
			uint uiSharedMemorySize=0;


			 
			dim3 BlkInGridConfig(iNumBlocks,1);
			dim3 ThdInBlkConfig(iNumThreads,1); 
			gpuGpuHalfSampleMEKernel<<<BlkInGridConfig,ThdInBlkConfig, uiSharedMemorySize>>>(pHostGMDA->m_iPUWidth,
																						     pHostGMDA->m_iPUHeight,
																						     extWidth,
																						     extHeight,
 																							 srcPtr,
																							 pHostGMDA->m_pYOrgPU,
																						     pHostGMDA->m_uiCost,
																							 pHostGMDA->m_iCostScale,
																							 pHostGMDA->m_iMvPredx,
																							 pHostGMDA->m_iMvPredy,
																							 iStride,
																							 gpufilteredBlock);
			CUDART_CHECK( cudaDeviceSynchronize() );

 			GPU::MemOpt::TransferToHost<Pel*>(cpufilteredBlock, gpufilteredBlock,4*FILTER_BLOCK_STRIDE*FILTER_BLOCK_STRIDE);
 			GPU::MemOpt::FreeDeviceMem<Pel>(gpufilteredBlock);
 			delete [] cpufilteredBlock;
 		}
 		*/
	}

	namespace Kernel
	{

//#define RedFactor 2
#define RedFactorH 4
#define RedFactorV 4	
#define WLINC 1 //(WLINC < 4)
 		
		__global__ void gpuGpuFullBlockSearchKernel(Int iPUWidth, 
													Int iPUHeight,
													Int iSRWidth,
													const Pel* m_pYRefPU,
													const Pel* m_pYOrgPU,
													Int m_ilx,
													Int m_ity,
													UInt m_uiCost,
													Int  m_iCostScale,
													Int  m_iMvPredx,
													Int  m_iMvPredy,
													UInt *m_puiSadArray,
													Int iStride)
													//UInt uiSharedMemorySegment1Size)
		{ 
			
			extern  __shared__  Pel sPU[];
			extern  __shared__  volatile UInt uiPU[]; //Alias

			const Pel* pYSR; 
			const Pel* pYOrg;
	 		uint32_t uiSad=0;
	 		int x, y, z, w;//, k;
	  

	 		z = threadIdx.x/iSRWidth;
	 		w = iPUWidth / WLINC;

 	 		//pYSR  = m_pYRefPU + blockIdx.x * iStride + threadIdx.x;
 	 		pYSR  = m_pYRefPU + blockIdx.x * iStride + threadIdx.x%iSRWidth;
	 		pYOrg =	m_pYOrgPU;

	 		// 1) Threads in a block is allocated equal to the search width (128)
	 		// The number of pixel in a row of PU varies from 64 to 8. To reduce 
	 		// iteration needed to move PU, all threads within a thread block
	 		// are working in parallel to move multiple rows of PU. This might 
	 		// reduce memory coalescing for small PU.
	 		
	 		// 2) To increase occupancy by constraining shared memory per block, 
	 		// moving PU to shared memory is only applied on PUs that have width 
	 		// and height less than 64.

	 		// 3) Immediately following the PU transfer is the calculation of SAD
	 		// of all locations, carried out by each thread in the thread block. 
	 		// Due to shared memory scarcity, reference frame pixels are not moved
	 		// into shared memory.
	 		
	 		// 4) Workload increase (WLINC). When WLINC=1, one thread processes
	 		// SAD for one location. When WLINC>1, WLINC number of thread processes
	 		// one location. For example, if WLINC = 2, 0th thread and 128th thread
	 		// process the SAD for the first location, former processes column 0-32. 
	 		// latter process column 32-63. The motivation is to maintain memory 
	 		// coalescing. WLINC is for increasing GPU utilization for SR < +/-64.
	 		
	 		if(iPUHeight < 64 && iPUWidth < 64)  
	 		{
	 			//Move PU to shared memory
		 		for(y=0;y < iPUHeight; y+=(blockDim.x/iPUWidth)) 
		 		{
		 			if((threadIdx.x + y*iPUWidth) < iPUWidth*iPUHeight)
		 			{
		 				sPU[threadIdx.x + y*iPUWidth] = pYOrg[threadIdx.x%iPUWidth + (threadIdx.x/iPUWidth + y)*iStride];	
		 			}
		 		}
				__syncthreads();

				for(y=0;y < iPUHeight; y+=RedFactorV)
		 		{
		 			
		 			//T&E: Move next row in the pYSR into shared memory for fast access
		 			/*
		 			for(k=0;k<2;k++)
		 			{
		 				sPU[uiSharedMemorySegment1Size + threadIdx.x + k*(iPUWidth-1)] = pYSR[y*iStride + k*(iPUWidth-1)]; 
		 			}
		 			
		 			__syncthreads(); //Ensure all threads completed transfer to shared memory.

		 			for(x=0;x < w; x+=RedFactorH)	
		 			{
		 				uiSad += abs(sPU[x + y*iPUWidth + w * z] - sPU[uiSharedMemorySegment1Size + threadIdx.x + x + w * z]);
 			
		 			}
		 			*/
		 			
		 			
		 			for(x=0;x < w; x+=RedFactorH)	
		 			{
		 				uiSad += abs(sPU[x + y*iPUWidth + w * z] - pYSR[x + y*iStride + w * z]);
 			
		 			}
		 			
		 			
		 			
		 		}

	 		}
			else 
 			{	
 				//Cache not fit, so skip
 				for(y=0;y < iPUHeight; y+=RedFactorV)
		 		{
		 			
		 			//T&E: Move next row in the pYSR into shared memory for fast access
		 			/*
		 			for(k=0;k<2;k++)
		 			{
		 				sPU[threadIdx.x + k*(iPUWidth-1)] = pYSR[y*iStride + k*(iPUWidth-1)]; 
		 			}
		 			
		 			__syncthreads(); //Ensure all threads completed transfer to shared memory.

		 			for(x=0;x < w; x+=RedFactorH)	
		 			{
		 				uiSad += abs(pYOrg[x + y*iStride + w * z]- sPU[threadIdx.x + x + w * z]);
		 			}
		 			
		 			*/
		 			
		 			for(x=0;x < w; x+=RedFactorH)
 		 			{
		 				uiSad += abs(pYOrg[x + y*iStride + w * z] - pYSR[x + y*iStride + w * (z)]);

		 			}
		 			
		 			
		 		}
		 		
 			}	
		
			__syncthreads();

	 		
//===>(Calculate SAD)
		
 			// The following loop sums the partial SADs (as a result of WLINC)
 			// The difference between y==0 and else is in first iteration,
 			// "+=" can't be used because the memory content is non zero.

 			for(y=0;y<WLINC;y++)
 			{
 				if((z) == y)  
 				{
 					if(y==0)
 					{
 						if((RedFactorH <= iPUWidth) && (RedFactorV <= iPUHeight))
 						{
 							uiSad = uiSad*RedFactorH*RedFactorV;
 						}
 						else if((RedFactorH > iPUWidth) && (RedFactorV <= iPUHeight))
 						{
 							uiSad = uiSad*iPUWidth*RedFactorV;
 						} 
 						else if((RedFactorH <= iPUWidth) && (RedFactorV > iPUHeight))
 						{
 							uiSad = uiSad*RedFactorH*iPUHeight;
 						}
 						else
 						{
 							uiSad = uiSad*iPUWidth*iPUHeight;
 						}
 						
 						uiPU[threadIdx.x%iSRWidth] = uiSad ;
 					}
 					else
 					{	
 						if((RedFactorH <= iPUWidth) && (RedFactorV <= iPUHeight))
 						{
 							uiSad = uiSad*RedFactorH*RedFactorV;
 						}
 						else if((RedFactorH > iPUWidth) && (RedFactorV <= iPUHeight))
 						{
 							uiSad = uiSad*iPUWidth*RedFactorV;
 						} 
 						else if((RedFactorH <= iPUWidth) && (RedFactorV > iPUHeight))
 						{
 							uiSad = uiSad*RedFactorH*iPUHeight;
 						}
 						else
 						{
 							uiSad = uiSad*iPUWidth*iPUHeight;
 						}

 						uiPU[threadIdx.x%iSRWidth] += uiSad;
 					}
 				}
 				__syncthreads();
 			}

			// 1) Only ISRWidth number thread is working in following stage
			// because there are only iSRWidth number of locations in a row. 
			// Each location has different MV bit rate that needs to be 
			// calculated. 

			// 2) The uiTemp[] = abs()*2+1 calculation didn't fully respect 
			// the integrity of the origional CPU algorithm. 

 			if(threadIdx.x < iSRWidth)
	 		{
				UInt uiTotalMvBits=0;
				UInt uiTemp[2];
				
				//Calculate cost
				y = (((threadIdx.x + m_ilx) << m_iCostScale) - m_iMvPredx); 
				w = (((blockIdx.x + m_ity)  << m_iCostScale) - m_iMvPredy); 

				uiTemp[0] = abs(y) * 2 + 1; // +1 Didn't fully respect algorithm integrity
				uiTemp[1] = abs(w) * 2 + 1; // +1 Didn't fully respect algorithm integrity

				uiTemp[0] = 1+((int)log2((float)uiTemp[0]))*2;
				uiTemp[1] = 1+((int)log2((float)uiTemp[1]))*2;

				uiTotalMvBits = uiTemp[0] + uiTemp[1] + 2;

//===>(True bit estimate algorithm)
//===>(reserved for maximum SR +/-128 (256x256))
	 			

				// 1) This stores cost and its location index with in the same 32-bit segment.
				// Cost occupies bit 0-23 while threadIdx.x occupies bit 24-31 for maximum 
				// index of 256 or +/-128. This is feasible because cost rarely exceed 2^24 - 1. 
				// Even if it does, it is a likely to be bad location.
 
				uiPU[threadIdx.x] = uiPU[threadIdx.x] + (m_uiCost * uiTotalMvBits >> 16) + (threadIdx.x<<24);
				__syncthreads();
	 			
	 			// 1) Without launch a second kernel, threads finds minimal row-wise SAD in parallel
	 			// in reduction fashion. Part 1 reduces SAD to 32 row-wise minimal and Part 2 reduces
	 			// to a single row-wise minimal. The need for two stage is because threads within a 
	 			// warp does not require synchronization and thus may improve performance.

	 			// 2) Row-wise minimals are transferred back to CPU for final reduction. The reason
	 			// for doing this is purely the result of trial and error. This proves to be the
	 			// most efficient way of finding minimal SAD. Also, global minimal can't be found
	 			// without launch another kernel because thread blocks can't be synchronized with
	 			// each other.
			
				//Find row-wise minSAD Part 1
				w=iSRWidth/2;
	 			for(x=w;x>=32;x>>=1)
				{
					if(threadIdx.x < x)
					{
						if ((uiPU[threadIdx.x] & 0x00FFFFFF) >= (uiPU[threadIdx.x + x] & 0x00FFFFFF))
							uiPU[threadIdx.x] = uiPU[threadIdx.x + x]; 
					}
					__syncthreads();

				} 		
	 
	 			//Find row-wise minSAD Part 2
	 			if(threadIdx.x < 32) //One wrap, no syncthread needed
				{
				//#pragma unroll
					for(x=16;x>0;x>>=1)
					{
						if(threadIdx.x < x)
						{
						 	if ((uiPU[threadIdx.x] & 0x00FFFFFF) < (uiPU[threadIdx.x + x] & 0x00FFFFFF))
								uiPU[threadIdx.x] = uiPU[threadIdx.x];
							else
								uiPU[threadIdx.x] = uiPU[threadIdx.x + x]; 
						}

						if(threadIdx.x==0)
						{
							m_puiSadArray[blockIdx.x] = uiPU[0];
						}
 					}	

				}
			
//===>(better final reduction but does not work)
			}

			//__syncthreads();

		}

		/**
		 * pHostGMDA->m_ilx, m_irx, m_ity, and m_iby are 
		 * left,right, top and bottom coordinates of the 
		 * search window. This is used to calculate threads 
		 * and threadblock required in the search. 

		 * WLINC is for increasing threads within block 
		 * for small search sizes (<+/-64). +/-64 is 
		 * just enough to saturate GPU.  
		*/
		void gpuGpuFullBlockSearch(GpuMeDataAccess* pHostGMDA)
		{
			
			cudaError_t status;
			dim3 BlkInGridConfig(pHostGMDA->GetNumBlocks(),1);
			dim3 ThdInBlkConfig(pHostGMDA->GetNumThreads() * WLINC,1); 
			gpuGpuFullBlockSearchKernel<<<BlkInGridConfig,ThdInBlkConfig, pHostGMDA->GetSharedMemSize()>>>(pHostGMDA->GetPUWidth(), 
																											pHostGMDA->GetPUHeight(),
																											pHostGMDA->GetSearchRange() * 2,
																											pHostGMDA->GetYRefPU(),
																											pHostGMDA->GetYOrgPU(),
																											pHostGMDA->GetLx(),
																											pHostGMDA->GetTy(),
																											pHostGMDA->GetCost(),
																											pHostGMDA->GetCostScale(),
																											pHostGMDA->GetMvPredX(),
																											pHostGMDA->GetMvPredY(),
																											pHostGMDA->GetSadArray(),
																											pHostGMDA->GetLStride());
																								//uiSharedMemorySegment1Size/sizeof(Pel));
			CUDART_CHECK( cudaDeviceSynchronize() );
 			
		}
		
	}

}



