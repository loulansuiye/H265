#ifndef __TENCGPUSEARCH__
#define __TENCGPUSEARCH__
#include "TLibCommon/TypeDef.h"
#include "TEncMEs.h"
#include <stdio.h>
class GpuMeDataAccess;

namespace CPU
{

}

namespace GPU {
	namespace MemOpt {
		template<class T>
		void
		AllocDeviceMem(T& src, size_t len);

		template<class T>
		void
		AllocHostMem(T& src, size_t len);

		template<class T>
		void
		TransferToDevice(T& src, T& targ, size_t len);

		template<class T>
		void
		TransferToHost(T& src, T& targ, size_t len);

		template<class T>
		void
		FreeDeviceMem(T *src);
		
		template<class T>
		void
		FreeHostMem(T *src);
	}

	namespace InterpolationKernel {
		void 
		gpuGpuHalfSampleME(GpuMeDataAccess* pHostGMDA, Int iStride, Int iMvX, Int iMvY);

	}

	namespace Kernel {
		void 
		gpuGpuFullBlockSearch(GpuMeDataAccess* pHostGMDA);
	}
}

#endif


	/*
	//Test
	if(iPUHeight < 64 && iPUWidth < 64)
	{
		for(y=0;y < iPUHeight; y++)
		{
			for(x=0;x < iPUWidth/WLINC; x++)
			{
				uiSad = sPU[x + y*iPUWidth + iPUWidth / WLINC * (threadIdx.x/iSRWidth)]; 
				uiSad = pYSR[x + y*iStride + iPUWidth / WLINC * (threadIdx.x/iSRWidth)];
			}
		}
	}
	else 
	{
		for(y=0;y < iPUHeight; y+=RedFactor)
		{
			for(x=0;x < iPUWidth/WLINC; x+=RedFactor)
			{
				uiSad = pYOrg[x + y*iStride + iPUWidth / WLINC * (threadIdx.x/iSRWidth)]; 
				uiSad = pYSR[x + y*iStride + iPUWidth / WLINC * (threadIdx.x/iSRWidth)];
			}
		}
	}	
	//~Test
	*/
  /*
  //Baseline benchmarking

  Distortion  uiSad;
  Distortion  uiSadBest = std::numeric_limits<Distortion>::max();
  Int         iBestX = 0;
  Int         iBestY = 0;
  Pel*  piRefSrch;

  //-- jclee for using the SAD function pointer
  m_pcRdCost->setDistParam( pcPatternKey, piRefY, iRefStride,  m_cDistParam );

  // fast encoder decision: use subsampled SAD for integer ME
  if ( m_pcEncCfg->getUseFastEnc() )
  {
    if ( m_cDistParam.iRows > 8 )
    {
      m_cDistParam.iSubShift = 1;
    }
  }

  int x=0; int y=0;
  piRefSrch = piRefY + x;
  m_cDistParam.pCur = piRefSrch;

  setDistParamComp(MAX_NUM_COMPONENT);

  m_cDistParam.bitDepth = pcPatternKey->getBitDepthY();
  //JCY complexity resume
  //m_cDistParam.iSubShift = 0;
  uiSad = m_cDistParam.DistFunc( &m_cDistParam );

  // motion cost
  uiSad += m_pcRdCost->getCost( x, y );
          
  uiSadBest = uiSad;
  iBestX    = x;
  iBestY    = y;
 
  rcMv.set( iBestX, iBestY );

  ruiSAD = uiSadBest - m_pcRdCost->getCost( iBestX, iBestY );
 */

/*
	//Debug
	m_puiSadArray[blockIdx.x * blockDim.x + threadIdx.x] = uiSad + (m_uiCost * uiTotalMvBits >> 16); 
	//~Debug


	//Debug
	GPU::MemOpt::TransferToHost<UInt*>(pHostGMDA->m_puiSadArray,puiRowMinSAD,sizeof(UInt)*iNumBlocks*iNumThreads);		
	for(int y=0;y<iNumBlocks;y++)
	{
		for(int x=0;x<iNumThreads;x++)
		{
			if(puiRowMinSAD[x + y*iNumThreads] < uiMinSAD)
			{
				uiMinSAD = puiRowMinSAD[x + y*iNumThreads];
				pHostGMDA->m_uiSad = uiMinSAD;
				pHostGMDA->m_iMvX = x + pHostGMDA->m_ilx + pHostGMDA->m_iLConstrain ;
				pHostGMDA->m_iMvY = y + pHostGMDA->m_ity + pHostGMDA->m_iTConstrain ;
			}
		}
	}

	//~Debug
*/

/*
  //Debug
      UInt* tmp;
      tmp = (UInt*)malloc(sizeof(UInt)*iNumBlocks*iNumThreads);
      GPU::MemOpt::TransferToHost<UInt*>(pHostGMDA->m_puiSadArray, tmp, sizeof(UInt)*iNumBlocks*iNumThreads);
      for(int i=0;i< iNumBlocks*iNumThreads;i++)
      {
      	if(i%64==0)
      	printf("\n");
        printf("%u ",tmp[i]);

      }
      free(tmp);
      printf("\n");
      //~Debug

*/


/*
	//Not 100% correct
	----------------------------------------------------------------------------------------------------------------------------------
	int iSharedMemorySize = pHostGMDA->m_iPUWidth * pHostGMDA->m_iPUHeight + 2*(pHostGMDA->m_iSearchRange + pHostGMDA->m_iPUWidth - 1)*pHostGMDA->m_iPUHeight; 
	----------------------------------------------------------------------------------------------------------------------------------
 	
 	pYSR = pDevGMDA->m_pYRefPU + blockIdx.x * iStride;

	//Move stripe to shared mmemory
	x = 2 * pDevGMDA->m_iSearchRange +  pDevGMDA->m_iPUWidth - 1;//#Pixel per row
	for(y=0;y< iPUHeight;y++)
	{

		sPU[iPUHeight*iPUWidth + threadIdx.x + y * x ] = pYSR[threadIdx.x + y*iStride];

		if(threadIdx.x < iPUWidth)
		{
			sPU[iPUHeight*iPUWidth + threadIdx.x + 2*pDevGMDA->m_iSearchRange - 1 + y * x] = pYSR[2*pDevGMDA->m_iSearchRange - 1 + y*iStride];
		}

	}

	__syncthreads();

	iStride = x;
	
	if(iPUHeight > 8 && y%RedFactor==0 && x%RedFactor==0)
	 	uiSad += abs(sPU[x + y*iPUWidth] - sPU[iPUWidth*iPUHeight + x + y*iStride]);

	if(iPUHeight <= 8)
	 	uiSad += abs(sPU[x + y*iPUWidth] - sPU[iPUWidth*iPUHeight + x + y*iStride]);


*/

/*
	if(iPUHeight > 8 && y%RedFactor==0 && x%RedFactor==0)
		uiSad += abs(sPU[x + y*iPUWidth] - pYSR[x + y*iStride]);

	if(iPUHeight <= 8)
		uiSad += abs(sPU[x + y*iPUWidth] - pYSR[x + y*iStride]);
*/

/*
	if(threadIdx.x==0)
	{
		for(x=0;x<blockDim.x;x++)
		{
			if ((uiPU[x] & 0x00FFFFFF) < (uiPU[0] & 0x00FFFFFF))
				uiPU[0] = uiPU[x];
		}
		pDevGMDA->m_puiSadArray[blockIdx.x] = uiPU[0];
	}
	__syncthreads();
			
*/

	
				/*
			 	if(threadIdx.x==0) //This can be better. For some reason, it doesn't work.
				{
					for(x=0;x<32;x++)
					{
						if ((uiPU[x] & 0x00FFFFFF) < (uiPU[0] & 0x00FFFFFF))
							uiPU[0] = uiPU[x];
					}
					m_puiSadArray[blockIdx.x] = uiPU[0];
				}
				*/
				
/*
	//Find row-wise minSad Part 2
	if(threadIdx.x < 32) //save some time
	{
	//#pragma unroll
		for(x=32;x>0;x>>=1)
		{
		 	if ((uiPU[threadIdx.x] & 0x00FFFFFF) < (uiPU[threadIdx.x + x] & 0x00FFFFFF))
				uiPU[threadIdx.x] = uiPU[threadIdx.x];
			else
				uiPU[threadIdx.x] = uiPU[threadIdx.x + x]; 
		}	

	}			

	if(threadIdx.x==0)
			pDevGMDA->m_puiSadArray[blockIdx.x] = uiPU[0];			
 */

		
 /*
		//JCY: expand motion cost calculation process
		UInt uiCost     = m_pcRdCost->getCost();
		Int  iCostScale = m_pcRdCost->getCostScale();
		TComMv cMvPred  = m_pcRdCost->getMvPred();
		UInt uiTotalMvBits=0;

		for(int hv=0;hv<2;hv++)
		{
		Int iVal;
		if(hv==0)
		{
		  iVal = ((x << iCostScale) - cMvPred.getHor());
		}
		else
		{
		  iVal = ((y << iCostScale) - cMvPred.getVer());
		}

		UInt uiLength = 1;
		UInt uiTemp = (iVal <= 0) ? (-iVal<<1)+1: (iVal<<1);

		while(1!=uiTemp)
		{
		  uiTemp >>=1;
		  uiLength+=2;
		}

		uiTotalMvBits += uiLength;
		}
		uiSad += uiCost * uiTotalMvBits >> 16;

		//~JCY
  */

