#include "TEncMEs.h"
#include <stdlib.h>
#include <limits>




void GpuMeDataAccess::Init() {
	Pel* pTmp;

	for(Int i=0; i < m_iGOPSize+1; i++)
	{
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetLPadSize());
		m_cYList.push_back(pTmp);

		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetCPadSize());
		m_cUList.push_back(pTmp);

		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetCPadSize());
		m_cVList.push_back(pTmp);

	}
	
	for(Int i=0; i < m_iGOPSize+1; i++)
	{
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetLPadSize());
		m_cYRefList0.push_back(pTmp);
		
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetCPadSize());
		m_cURefList0.push_back(pTmp);
		
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetCPadSize());
		m_cVRefList0.push_back(pTmp);
	}

	for(Int i=0; i < m_iGOPSize+1; i++)
	{
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetLPadSize());
		m_cYRefList1.push_back(pTmp);
		
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetCPadSize());
		m_cURefList1.push_back(pTmp);
		
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * GetCPadSize());
		m_cVRefList1.push_back(pTmp);
	}
	
	GPU::MemOpt::AllocDeviceMem<UInt*>(m_puiSadArray,sizeof(UInt) * 4 * m_iSearchRange * m_iSearchRange);
	GPU::MemOpt::AllocHostMem<UInt*>(m_puiSadArrayCPU,sizeof(UInt) * 4 * m_iSearchRange * m_iSearchRange);

	/*
	if( (m_puiSadArrayCPU = (UInt*)cudaMall(sizeof(UInt) * 4 * m_iSearchRange * m_iSearchRange)) == NULL)
	{
		perror("Allocating m_puiSadArrayCPU Failed\n");
	}
	*/
}

void GpuMeDataAccess::Destroy() {
	for (Int i=0; i < m_iGOPSize+1; i++) {
		GPU::MemOpt::FreeDeviceMem<Pel>(m_cYList[i]);
		GPU::MemOpt::FreeDeviceMem<Pel>(m_cUList[i]);
		GPU::MemOpt::FreeDeviceMem<Pel>(m_cVList[i]);
		GPU::MemOpt::FreeDeviceMem<Pel>(m_cYRefList0[i]);
		GPU::MemOpt::FreeDeviceMem<Pel>(m_cURefList0[i]);
		GPU::MemOpt::FreeDeviceMem<Pel>(m_cVRefList0[i]);
		GPU::MemOpt::FreeDeviceMem<Pel>(m_cYRefList1[i]);
		GPU::MemOpt::FreeDeviceMem<Pel>(m_cURefList1[i]);
		GPU::MemOpt::FreeDeviceMem<Pel>(m_cVRefList1[i]);
	}

	GPU::MemOpt::FreeDeviceMem<UInt>(m_puiSadArray);
	GPU::MemOpt::FreeHostMem<UInt>(m_puiSadArrayCPU);
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
	if (iList==0) {
	    GPU::MemOpt::TransferToDevice<Pel*>(pYuv[0], m_cYRefList0[iIndex], GetLPadSize()*sizeof(Pel));
	    GPU::MemOpt::TransferToDevice<Pel*>(pYuv[1], m_cURefList0[iIndex], GetCPadSize()*sizeof(Pel));
	    GPU::MemOpt::TransferToDevice<Pel*>(pYuv[2], m_cVRefList0[iIndex], GetCPadSize()*sizeof(Pel));
	}
	else {
    	GPU::MemOpt::TransferToDevice<Pel*>(pYuv[0], m_cYRefList1[iIndex], GetLPadSize()*sizeof(Pel));
   	 	GPU::MemOpt::TransferToDevice<Pel*>(pYuv[1], m_cURefList1[iIndex], GetCPadSize()*sizeof(Pel));
    	GPU::MemOpt::TransferToDevice<Pel*>(pYuv[2], m_cVRefList1[iIndex], GetCPadSize()*sizeof(Pel));
	}

}

void GpuMeDataAccess::CopyCurFrameToGpu(Pel* pYuv[3]) {
	GPU::MemOpt::TransferToDevice<Pel*>( pYuv[0], m_cYList[0], GetLPadSize()*sizeof(Pel));
	GPU::MemOpt::TransferToDevice<Pel*>( pYuv[1], m_cUList[0], GetCPadSize()*sizeof(Pel));
	GPU::MemOpt::TransferToDevice<Pel*>( pYuv[2], m_cVList[0], GetCPadSize()*sizeof(Pel));
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

void GpuMeDataAccess::SetSearchWindowPtr() {

	if (m_iRefListId==0) {
		m_pYRefPU = m_cYRefList0[m_iRefId] + GetLOrgOffset() + m_iPUOffset   + m_ity   * GetLStride()   + m_ilx;
		m_pURefPU = m_cURefList0[m_iRefId] + GetCOrgOffset() + m_iPUOffset/4 + m_ity/2 * GetLStride()/2 + m_ilx/2;
		m_pVRefPU = m_cVRefList0[m_iRefId] + GetCOrgOffset() + m_iPUOffset/4 + m_ity/2 * GetLStride()/2 + m_ilx/2;
	} //Top-left for the PU of current SR
	else {
		m_pYRefPU = m_cYRefList1[m_iRefId] + GetLOrgOffset() + m_iPUOffset   + m_ity   * GetLStride()   + m_ilx;
		m_pURefPU = m_cURefList1[m_iRefId] + GetCOrgOffset() + m_iPUOffset/4 + m_ity/2 * GetLStride()/2 + m_ilx/2;
		m_pVRefPU = m_cVRefList1[m_iRefId] + GetCOrgOffset() + m_iPUOffset/4 + m_ity/2 * GetLStride()/2 + m_ilx/2;
	}//Top-left for the PU of current SR
}

void GpuMeDataAccess::SetPuPtr() {
	m_pYOrgPU = m_cYList[0] + GetLOrgOffset() + m_iPUOffset;
 	m_pUOrgPU = m_cUList[0] + GetCOrgOffset() + m_iPUOffset/4;
	m_pVOrgPU = m_cVList[0] + GetCOrgOffset() + m_iPUOffset/4;
  
}

void GpuMeDataAccess::GpuSearchWrapper() {

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

