#include "TEncMEs.h"
#include <stdlib.h>
#include <limits>




void GpuMeDataAccess::Init(bool bIsMainThread) {

	//!< Top switch for number of device. Replace by cudaGetDeviceCount();
	m_iNumGPU        = NUM_GPU;
	m_bIsMainThread = bIsMainThread;	

	Pel* pTmp;

	if (m_bIsMainThread==true) {
	 	m_cYList = new vector<Pel*>[m_iNumGPU];
		m_cUList = new vector<Pel*>[m_iNumGPU]; 
		m_cVList = new vector<Pel*>[m_iNumGPU];

		m_cYRefList0 = new vector<Pel*>[m_iNumGPU];
		m_cURefList0 = new vector<Pel*>[m_iNumGPU];
		m_cVRefList0 = new vector<Pel*>[m_iNumGPU];

		m_cYRefList1 = new vector<Pel*>[m_iNumGPU];
		m_cURefList1 = new vector<Pel*>[m_iNumGPU];
		m_cVRefList1 = new vector<Pel*>[m_iNumGPU];


		for (Int iDevice=0; iDevice < m_iNumGPU; iDevice++) {
			
			GPU::MemOpt::ChangeDevice(iDevice);

			for(Int i=0; i < m_iGOPSize+1; i++)
			{
				GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetLPadSize());
				m_cYList[iDevice].push_back(pTmp);

				GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetCPadSize());
				m_cUList[iDevice].push_back(pTmp);

				GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetCPadSize());
				m_cVList[iDevice].push_back(pTmp);

			}
			
			for(Int i=0; i < m_iGOPSize+1; i++)
			{
				GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetLPadSize());
				m_cYRefList0[iDevice].push_back(pTmp);
				
				GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetCPadSize());
				m_cURefList0[iDevice].push_back(pTmp);
				
				GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetCPadSize());
				m_cVRefList0[iDevice].push_back(pTmp);
			}

			for(Int i=0; i < m_iGOPSize+1; i++)
			{
				GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetLPadSize());
				m_cYRefList1[iDevice].push_back(pTmp);
				
				GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetCPadSize());
				m_cURefList1[iDevice].push_back(pTmp);
				
				GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetCPadSize());
				m_cVRefList1[iDevice].push_back(pTmp);
			}
		}
	}
	else {

		m_puiSadArrayDevice = new UInt*[m_iNumGPU];

		for (Int iDevice=0; iDevice < m_iNumGPU; iDevice++) {
			GPU::MemOpt::ChangeDevice(iDevice);
			GPU::MemOpt::AllocDeviceMem<UInt*>(m_puiSadArrayDevice[iDevice],sizeof(UInt) * 4 * m_iSearchRange * m_iSearchRange);
		}

		GPU::MemOpt::AllocHostMem<UInt*>(m_puiSadArrayCPU,sizeof(UInt) * 4 * m_iSearchRange * m_iSearchRange);
	} //!< Non-main thread only allocates SAD buffer for itself on two devices.
	
	 
}

void GpuMeDataAccess::Destroy() {
	if (m_bIsMainThread) {
		for (Int iDevice = 0; iDevice < m_iNumGPU; iDevice++) {
			for (Int i=0; i < m_iGOPSize+1; i++) {
				GPU::MemOpt::FreeDeviceMem<Pel>(m_cYList[iDevice][i]);
				GPU::MemOpt::FreeDeviceMem<Pel>(m_cUList[iDevice][i]);
				GPU::MemOpt::FreeDeviceMem<Pel>(m_cVList[iDevice][i]);
				GPU::MemOpt::FreeDeviceMem<Pel>(m_cYRefList0[iDevice][i]);
				GPU::MemOpt::FreeDeviceMem<Pel>(m_cURefList0[iDevice][i]);
				GPU::MemOpt::FreeDeviceMem<Pel>(m_cVRefList0[iDevice][i]);
				GPU::MemOpt::FreeDeviceMem<Pel>(m_cYRefList1[iDevice][i]);
				GPU::MemOpt::FreeDeviceMem<Pel>(m_cURefList1[iDevice][i]);
				GPU::MemOpt::FreeDeviceMem<Pel>(m_cVRefList1[iDevice][i]);
			}
		}

		
		delete [] m_cYList;
		delete [] m_cUList;
		delete [] m_cVList;

		delete [] m_cYRefList0;
		delete [] m_cURefList0;
		delete [] m_cVRefList0;

		delete [] m_cYRefList1;
		delete [] m_cURefList1;
		delete [] m_cVRefList1;	
	}
	else { 
		for (Int iDevice = 0; iDevice < m_iNumGPU; iDevice++) {
			GPU::MemOpt::FreeDeviceMem<UInt>(m_puiSadArrayDevice[iDevice]);
		}

		GPU::MemOpt::FreeHostMem<UInt>(m_puiSadArrayCPU);
		delete [] m_puiSadArrayDevice;
	}
}

void GpuMeDataAccess::PreparePU(Int iPUOffset, Int iPUWidth, Int iPUHeight, Int iRefListId, Int iRefId, Int iMvPredx, Int iMvPredy, UInt uiCost, Int iCostScale) {
	m_iPUOffset = iPUOffset;
	m_iPUHeight = iPUHeight;
	m_iPUWidth  = iPUWidth;
	m_iRefListId = iRefListId;
	m_iRefId    = iRefId;
	m_iMvPredx  = iMvPredx;
	m_iMvPredy  = iMvPredy;
	m_uiCost    = uiCost;
	m_iCostScale = iCostScale;

	  
   
}

void GpuMeDataAccess::InitSearchData(Int iSearchRange = 128, Int iBiPredSearchRange = 16, Int iFastSearch = 0) {
	m_iSearchRange = iSearchRange; 
	m_biPredSearchRange = iBiPredSearchRange;
	m_iFastSearch = iFastSearch;
    m_ilx = -iSearchRange; 
    m_irx = iSearchRange; 
    m_ity = -iSearchRange; 
    m_iby = iSearchRange;
}

void GpuMeDataAccess::CopyRefFrameToGpuByIndex(Pel* pYuv[3], Int iIndex, Int iList) {
	
	for (int iDevice = 0; iDevice < m_iNumGPU; iDevice++) {

		GPU::MemOpt::ChangeDevice(iDevice);

		if (iList==0) {
		    GPU::MemOpt::TransferToDevice<Pel*>(pYuv[0], m_cYRefList0[iDevice][iIndex], GetLPadSize()*sizeof(Pel));
		    GPU::MemOpt::TransferToDevice<Pel*>(pYuv[1], m_cURefList0[iDevice][iIndex], GetCPadSize()*sizeof(Pel));
		    GPU::MemOpt::TransferToDevice<Pel*>(pYuv[2], m_cVRefList0[iDevice][iIndex], GetCPadSize()*sizeof(Pel));
		}
		else {
	    	GPU::MemOpt::TransferToDevice<Pel*>(pYuv[0], m_cYRefList1[iDevice][iIndex], GetLPadSize()*sizeof(Pel));
	   	 	GPU::MemOpt::TransferToDevice<Pel*>(pYuv[1], m_cURefList1[iDevice][iIndex], GetCPadSize()*sizeof(Pel));
	    	GPU::MemOpt::TransferToDevice<Pel*>(pYuv[2], m_cVRefList1[iDevice][iIndex], GetCPadSize()*sizeof(Pel));
		}
	}
}

void GpuMeDataAccess::CopyCurFrameToGpu(Pel* pYuv[3]) {
	for (int iDevice = 0; iDevice < m_iNumGPU; iDevice++) {
		GPU::MemOpt::ChangeDevice(iDevice);
		GPU::MemOpt::TransferToDevice<Pel*>( pYuv[0], m_cYList[iDevice][0], GetLPadSize()*sizeof(Pel));
		GPU::MemOpt::TransferToDevice<Pel*>( pYuv[1], m_cUList[iDevice][0], GetCPadSize()*sizeof(Pel));
		GPU::MemOpt::TransferToDevice<Pel*>( pYuv[2], m_cVList[iDevice][0], GetCPadSize()*sizeof(Pel));
	}
}

void GpuMeDataAccess::SetSearchWindowSize(Int iSrchRngHorLeft, Int iSrchRngHorRight, Int iSrchRngVerTop, Int iSrchRngVerBottom) {

	m_ilx = iSrchRngHorLeft;
    m_irx = iSrchRngHorRight;
    m_ity = iSrchRngVerTop;
    m_iby = iSrchRngVerBottom;
      
	//Fixing rcMvPred out-of-bound/Clipping problem (May affect RD performance)
	//Force search width to 2^N
	if ((-m_ilx + m_irx) != m_iSearchRange * 2) {
		m_ilx = -m_iSearchRange;
		m_irx =  m_iSearchRange;
	}


	//search height forcing is optional
	//if((-m_ity + m_iby) != m_iSearchRange * 2)
	//{
	//   m_ity = -m_iSearchRange;
	//   m_iby =  m_iSearchRange;
	//}

	//Test
	//m_ity = -m_iSearchRange + 32;
	//m_iby = 32;
	//~Test
}

//!< Only non-main threads performs Search
void GpuMeDataAccess::SetSearchWindowPtr() {

	if (m_iRefListId==0) {
		m_pYRefPU = m_cYRefList0[m_iGpuId][m_iRefId] + GetLOrgOffset() + m_iPUOffset   + m_ity   * GetLStride()   + m_ilx;
		m_pURefPU = m_cURefList0[m_iGpuId][m_iRefId] + GetCOrgOffset() + m_iPUOffset/4 + m_ity/2 * GetLStride()/2 + m_ilx/2;
		m_pVRefPU = m_cVRefList0[m_iGpuId][m_iRefId] + GetCOrgOffset() + m_iPUOffset/4 + m_ity/2 * GetLStride()/2 + m_ilx/2;
	} //Top-left for the PU of current SR
	else {
		m_pYRefPU = m_cYRefList1[m_iGpuId][m_iRefId] + GetLOrgOffset() + m_iPUOffset   + m_ity   * GetLStride()   + m_ilx;
		m_pURefPU = m_cURefList1[m_iGpuId][m_iRefId] + GetCOrgOffset() + m_iPUOffset/4 + m_ity/2 * GetLStride()/2 + m_ilx/2;
		m_pVRefPU = m_cVRefList1[m_iGpuId][m_iRefId] + GetCOrgOffset() + m_iPUOffset/4 + m_ity/2 * GetLStride()/2 + m_ilx/2;
	}//Top-left for the PU of current SR
}

void GpuMeDataAccess::SetPuPtr() {
	m_pYOrgPU = m_cYList[m_iGpuId][0] + GetLOrgOffset() + m_iPUOffset;
 	m_pUOrgPU = m_cUList[m_iGpuId][0] + GetCOrgOffset() + m_iPUOffset/4;
	m_pVOrgPU = m_cVList[m_iGpuId][0] + GetCOrgOffset() + m_iPUOffset/4;
  
}

void GpuMeDataAccess::GpuSearchWrapper() {
	/** Change device has effective until the end
	 *  of the thread life. Device for each thread
	 *  is known when thread id is known. Thus
	 *  ChangeDevice can be moved to much higher
	 *  level so to reduce ChangDevice calls.
	 */

	GPU::MemOpt::ChangeDevice(m_iGpuId);

	SetKernelParameter();
	GPU::Kernel::gpuGpuFullBlockSearch(this);
	PostKernelProcessing();
}

/**
 * Shared memory is used for storing PU and SADs 
 * at different stages inside kernel. Thus, it's 
 * size must be allocated to whichever is the largest. 
 */
void GpuMeDataAccess::SetKernelParameter() {

	Int	iSharedMemorySegment1Size = 0;

	m_iNumThreads                = -GetLx() + GetRx();
	m_iNumBlocks                 = -GetTy() + GetBy();
	
	// Segment1 is for storing PU/SADs
	if(m_iPUWidth < 64 && m_iPUHeight < 64)
		iSharedMemorySegment1Size = GetPUWidth() * GetPUHeight() * sizeof(Pel); 

	if(iSharedMemorySegment1Size < 2 * GetSearchRange() * sizeof(UInt))
		iSharedMemorySegment1Size = 2 * GetSearchRange() * sizeof(UInt); //For SAD
	
	// Segment2 is for storing a row of search region pixel
	// T&E: 1) Reserve 2*SearchRange*sizeof(Pel) in the shared memory.
	//         It does after already allocated shared memory. 
	//      2) Offset is uiSharedMemorySegment1Size/sizeof(Pel) because 
	//         This is the number of pel to be skipped. Remember ptr+N
	//         depends on type size.
	m_iSharedMemSize = iSharedMemorySegment1Size;// + (2*pHostGMDA->m_iSearchRange + pHostGMDA->m_iPUWidth - 1) * sizeof(Pel);	
}

/**
 * After kernel follows, the minimization of 
 * row-wise minimals to a single global minimal.
 * Cost and its location index are encoded 
 * together in a UInt type, where cost occupies
 * bit 0-23 and index occupies bit 24-31. 
 */ 
void GpuMeDataAccess::PostKernelProcessing() {
	UInt uiMinSAD=std::numeric_limits<UInt>::max();
	UInt *puiRowMinSAD = m_puiSadArrayCPU;

	//This transfer took 8 seconds in 9 frame
	GPU::MemOpt::TransferToHost<UInt*>(m_puiSadArray, puiRowMinSAD, sizeof(UInt)*m_iNumBlocks);		

	for (int i=0;i< m_iNumBlocks;i++)
	{
		if((puiRowMinSAD[i] & 0x00FFFFFF) < (uiMinSAD & 0x00FFFFFF))
		{
			uiMinSAD = puiRowMinSAD[i];
			m_uiSad = uiMinSAD & 0x00FFFFFF;
			m_iMvX  = ((uiMinSAD & 0xFF000000)>>24) + m_ilx + m_iLConstrain ;
			m_iMvY  = i + m_ity + m_iTConstrain ;
		}
	}

}


void GpuMeDataAccess::SetGpuId(Int iGpuId) {
	m_iGpuId = iGpuId % m_iNumGPU;
}

void GpuMeDataAccess::UpdateWorkingThread(GpuMeDataAccess* pMainThreadGpuDataAccess) {
	m_cYRefList0 = pMainThreadGpuDataAccess->GetYRefList0();
	m_cURefList0 = pMainThreadGpuDataAccess->GetURefList0();
	m_cVRefList0 = pMainThreadGpuDataAccess->GetVRefList0();

	m_cYRefList1 = pMainThreadGpuDataAccess->GetYRefList1();
	m_cURefList1 = pMainThreadGpuDataAccess->GetURefList1();
	m_cVRefList1 = pMainThreadGpuDataAccess->GetVRefList1();

	m_cYList = pMainThreadGpuDataAccess->GetYList();
	m_cUList = pMainThreadGpuDataAccess->GetUList();
	m_cVList = pMainThreadGpuDataAccess->GetVList();

	m_puiSadArray = m_puiSadArrayDevice[m_iGpuId];
}


Int GpuMeDataAccess::GetFrameWidth() const {
	return m_iFrameWidth;
}

Int GpuMeDataAccess::GetFrameHeight() const {
	return m_iFrameHeight;
}

Int GpuMeDataAccess::GetLPadSize() const { 
	return (m_iFrameWidth + 2*m_iPadlen) * (m_iFrameHeight + 2*m_iPadlen);
}

Int GpuMeDataAccess::GetCPadSize() const { 
	return (m_iFrameWidth/2 + 2*m_iPadlen/2) * (m_iFrameHeight/2 + 2*m_iPadlen/2);
}

Int GpuMeDataAccess::GetLStride() const {
	return m_iFrameWidth + 2*m_iPadlen;
}

Int GpuMeDataAccess::GetCStride() const {
	return m_iFrameWidth/2 + 2*m_iPadlen/2;
}

Int GpuMeDataAccess::GetLOrgOffset() const {
	return m_iPadlen   * GetLStride() + m_iPadlen;
}

Int GpuMeDataAccess::GetCOrgOffset() const {
	return m_iPadlen/2 * GetCStride() + m_iPadlen/2;
}

Int GpuMeDataAccess::GetPUWidth() const {
	return m_iPUWidth;
}

Int GpuMeDataAccess::GetPUHeight() const {
	return m_iPUHeight;
}

Int GpuMeDataAccess::GetMvX() const {
	return m_iMvX;
}

Int GpuMeDataAccess::GetMvY() const {
	return m_iMvY;
}

UInt GpuMeDataAccess::GetSad() const {
	return m_uiSad;
}

Int GpuMeDataAccess::GetLx() const {
	return m_ilx;
}

Int GpuMeDataAccess::GetRx() const {
	return m_irx;
}

Int GpuMeDataAccess::GetTy() const {
	return m_ity;
}

Int GpuMeDataAccess::GetBy() const {
	return m_iby;

}

Int GpuMeDataAccess::GetSearchRange() const {
	return m_iSearchRange;
}

UInt GpuMeDataAccess::GetCost() const {
	return m_uiCost;
}

Int GpuMeDataAccess::GetCostScale() const {
	return m_iCostScale;
}

Int GpuMeDataAccess::GetMvPredX() const {
	return m_iMvPredx;
}

Int  GpuMeDataAccess::GetMvPredY() const {
	return m_iMvPredy;
}

Int GpuMeDataAccess::GetNumThreads() const {
	return m_iNumThreads;
}

Int GpuMeDataAccess::GetNumBlocks() const {
	return m_iNumBlocks;
}

Int GpuMeDataAccess::GetSharedMemSize() const {
	return m_iSharedMemSize;
}

const Pel* GpuMeDataAccess::GetYRefPU() const {
	return m_pYRefPU;
}	
const Pel* GpuMeDataAccess::GetYOrgPU() const {
	return m_pYOrgPU;
}

UInt* GpuMeDataAccess::GetSadArray() const {
	return m_puiSadArray;  
}

	
vector<Pel*>* GpuMeDataAccess::GetYList() const {
	return m_cYList;
}

vector<Pel*>* GpuMeDataAccess::GetUList() const {
	return m_cUList;
}

vector<Pel*>* GpuMeDataAccess::GetVList() const {
	return m_cVList;
}

vector<Pel*>* GpuMeDataAccess::GetYRefList0() const {
	return m_cYRefList0;
}

vector<Pel*>* GpuMeDataAccess::GetURefList0() const {
	return m_cURefList0;
}

vector<Pel*>* GpuMeDataAccess::GetVRefList0() const {
	return m_cVRefList0;
}

vector<Pel*>* GpuMeDataAccess::GetYRefList1() const {
	return m_cYRefList1;
}

vector<Pel*>* GpuMeDataAccess::GetURefList1() const {
	return m_cURefList1;
}

vector<Pel*>* GpuMeDataAccess::GetVRefList1() const {
	return m_cVRefList1;

}
