#ifndef __TENCMES__
#define __TENCMES__
#include "TEncGPUSearch.h"
#include "TLibCommon/TypeDef.h"
#include "TLibCommon/CommonDef.h"

#include <vector>

using std::vector;

class GpuMeDataAccess {	
public:
	GpuMeDataAccess(Int iFrameWidth = 640, Int iFrameHeight = 480, Int iGOPSize = 8):
																					m_iPUWidth(0),
																					m_iPUHeight(0),
																					m_iFrameWidth(iFrameWidth),
																					m_iFrameHeight(iFrameHeight),
																					m_iGOPSize(iGOPSize),
																					m_iPUId(0),			   
																					m_iPadlen(80),
																					m_iRefListId(0),
																					m_iRefId(0),
																					m_iLConstrain(0),
																					m_iRConstrain(0),
																					m_iTConstrain(0),
																					m_iBConstrain(0),
																					m_iNumGPU(1)
																					{}

	~GpuMeDataAccess(){}
	void Init(bool bIsMainThread);
	void Destroy();
	void InitSearchData(Int iSearchRange, Int iBiPredSearchRange, Int iFastSearch);
	void PreparePU(Int iPUOffset, Int iPUWidth, Int iPUHeight, Int iRefListId, Int iRefId, Int iMvPredx, Int iMvPredy, UInt uiCost, Int iCostScale);

	void CopyRefFrameToGpuByIndex(Pel* pYuv[3],Int iIndex, Int iList);
	void CopyCurFrameToGpu(Pel* pYuv[3]);

	void SetSearchWindowSize(Int iSrchRngHorLeft, Int iSrchRngHorRight, Int iSrchRngVerTop, Int iSrchRngVerBottom);
	void SetSearchWindowPtr();
	void SetPuPtr();
	void GpuSearchWrapper();
	void SetKernelParameter();
	void PostKernelProcessing();
	void SetGpuId(Int iGpuId);
	void UpdateWorkingThread(GpuMeDataAccess* pMainThreadGpuDataAccess);

	Int GetFrameWidth() const;
	Int GetFrameHeight() const;
	
 	Int GetLPadSize() const;
	Int GetCPadSize() const;
	Int GetLStride() const;
	Int GetCStride() const;
	Int GetLOrgOffset() const;
	Int GetCOrgOffset() const;
	Int GetPUWidth() const;
	Int GetPUHeight() const;
	Int GetMvX() const;
	Int GetMvY() const;
	UInt GetSad() const;

	Int GetLx() const;
	Int GetRx() const;
	Int GetTy() const;
	Int GetBy() const;
	Int GetSearchRange() const;
	UInt GetCost() const;
	Int GetCostScale() const;
	Int GetMvPredX() const;
	Int GetMvPredY() const;

	Int GetNumThreads() const;
	Int GetNumBlocks() const;
	Int GetSharedMemSize() const;

	const Pel*  GetYRefPU() const; 	
	const Pel*  GetYOrgPU() const; 	
	
	vector<Pel*>*   GetYList() const;
	vector<Pel*>*   GetUList() const;
	vector<Pel*>*   GetVList() const;

	vector<Pel*>*   GetYRefList0() const;
	vector<Pel*>*   GetURefList0() const;
	vector<Pel*>*   GetVRefList0() const;

	vector<Pel*>*   GetYRefList1() const;
	vector<Pel*>*   GetURefList1() const;
	vector<Pel*>*   GetVRefList1() const;

	UInt*  GetSadArray() const;

private:

	Int m_iPUWidth;
	Int m_iPUHeight;
	Int m_iFrameWidth;
	Int m_iFrameHeight;
  	Int m_iGOPSize;
	Int m_iPUId;
	Int m_iPadlen;
	Int m_iRefListId;
	Int m_iRefId;
	Int m_iLConstrain;
	Int m_iRConstrain;
	Int m_iTConstrain;
	Int m_iBConstrain;

	Int m_ilx;
	Int m_irx;
	Int m_ity;
	Int m_iby;

	Int m_iPUOffset;


	vector<Pel*> *m_cYList;
	vector<Pel*> *m_cUList;
	vector<Pel*> *m_cVList;

	Pel*   m_pYOrgPU;
	Pel*   m_pUOrgPU;
	Pel*   m_pVOrgPU;
 

	vector<Pel*> *m_cYRefList0;
	vector<Pel*> *m_cURefList0;
	vector<Pel*> *m_cVRefList0;

	vector<Pel*> *m_cYRefList1;
	vector<Pel*> *m_cURefList1;
	vector<Pel*> *m_cVRefList1;

	Pel*   m_pYRefPU;
	Pel*   m_pURefPU;
	Pel*   m_pVRefPU;

	Int m_iMvPredx;
	Int m_iMvPredy;
	Int m_iCostScale;
	UInt m_uiCost;
	UInt*  m_puiSadArray;
	UInt** m_puiSadArrayDevice;
	UInt*  m_puiSadArrayCPU;
	
	Int m_iSearchRange;
  	Int m_biPredSearchRange; 
  	Int m_iFastSearch;

  	Int m_iMvX;
  	Int m_iMvY;
  	UInt m_uiSad;
	
 	//!< Kernel Parameters	
  	Int m_iNumGPU;
  	Int m_iNumThreads;
  	Int m_iNumBlocks;
  	Int m_iSharedMemSize;
  	Int m_iGpuId;

  	//!< Multithreading parameters
  	bool m_bIsMainThread;

};

#endif

