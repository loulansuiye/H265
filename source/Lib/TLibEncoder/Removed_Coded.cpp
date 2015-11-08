

/*
//===>(True bit estimate algorithm)
-------------------------------------
				UInt uiLength;
				UInt uiTemp;
				for(x=0;x<2;x++)
				{

					if(x==0)
					{
						y = (((threadIdx.x + m_ilx) << m_iCostScale) - m_iMvPredx); 
					}
					else
					{
						y = (((blockIdx.x + m_ity)  << m_iCostScale) - m_iMvPredy); 
					}

					//1. HEVC 
					uiTemp = (y <= 0) ? (-y<<1)+1: (y<<1);

					//2. JCY No divergency
					//uiTemp = abs(y) * 2 + 1;

					//1. HEVC 
					uiLength = 1;
					while(1!=uiTemp)
					{
						uiTemp >>=1;
						uiLength+=2;
					}
					uiTotalMvBits += uiLength;
					
					//2. JCY No divergency
					//uiTotalMvBits += 1+((int)log2((float)uiTemp))*2;
				}
	 			
				__syncthreads();
*/


/*
//===>(reserved for maximum SR +/-128 (256x256))
				//reserved for maximum SR +/-128 (256x256)
				
				if(iPUHeight < 64 && iPUWidth < 64)
		 			uiPU[threadIdx.x] = uiSad + (m_uiCost * uiTotalMvBits >> 16) + (threadIdx.x<<24);
		 		else
					uiPU[threadIdx.x] = uiSad*RedFactor*RedFactor + (m_uiCost * uiTotalMvBits >> 16) + (threadIdx.x<<24);
				

				
				if(iPUHeight < 64 && iPUWidth < 64)
		 			uiPU[threadIdx.x] = uiPU[threadIdx.x]+ (m_uiCost * uiTotalMvBits >> 16) + (threadIdx.x<<24);
		 		else
					uiPU[threadIdx.x] = uiPU[threadIdx.x] + (m_uiCost * uiTotalMvBits >> 16) + (threadIdx.x<<24);
				
*/


/*
//===>(better final reduction but does not work)
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
//===>(Calculate SAD)

				if(iPUHeight < 64 && iPUWidth < 64)
				{
					for(y=0;y < iPUHeight; y++)
					{
						for(x=0;x < iPUWidth; x++)
						{
							uiSad += abs(sPU[x + y*iPUWidth] - pYSR[x + y*iStride]);
						}
					}
				}
				else 
				{
					for(y=0;y < iPUHeight; y+=RedFactorV)
					{
						for(x=0;x < iPUWidth; x+=RedFactorH)
						{
							uiSad += abs(pYOrg[x + y*iStride] - pYSR[x + y*iStride]);
						}
					}
				}
*/
