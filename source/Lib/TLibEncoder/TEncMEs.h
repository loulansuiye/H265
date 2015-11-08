#ifndef __TENCMES__
#define __TENCMES__
#include "TEncGPUSearch.h"
#include "TLibCommon/TypeDef.h"
#include <vector>

class CpuMeDataAccess
{	
public:
	CpuMeDataAccess(Int iFrameWidth = 640, Int iFrameHeight = 480, Int iGOPSize = 8):
	m_iPUWidth(0),
	m_iPUHeight(0)
	{				   
		m_iFrameWidth = iFrameWidth;
		m_iFrameHeight = iFrameHeight;
		m_iGOPSize = iGOPSize;
		m_iPUId = 0;				   
		m_iPadlen = 80;
		m_iRefListId = 0;
		m_iRefId = 0;
		m_iLConstrain = 0;
		m_iRConstrain = 0;
		m_iTConstrain = 0;
		m_iBConstrain = 0;
		//m_pYCurPU = 0;
		//m_pUCurPU = 0;
		//m_pVCurPU = 0;
		 
	}
	virtual ~CpuMeDataAccess(){}
	virtual void init();
	void initSearchData(Int iSearchRange = 128, Int iBiPredSearchRange = 16, Int iFastSearch = 0)
								{m_iSearchRange = iSearchRange; m_biPredSearchRange = iBiPredSearchRange;m_iFastSearch = iFastSearch;
								 m_ilx = -iSearchRange; m_irx = iSearchRange; m_ity = -iSearchRange; m_iby = iSearchRange;}
	virtual void destroy();
	
	void preparePU(Int iPUOffset, Int iPUWidth, Int iPUHeight, Int iRefListId, Int iRefId, Int iMvPredx, Int iMvPredy, UInt uiCost, Int iCostScale)
	{
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

	Int getLPadSize(){ return (m_iFrameWidth + 2*m_iPadlen) * (m_iFrameHeight + 2*m_iPadlen);}
	Int getCPadSize(){ return (m_iFrameWidth/2 + 2*m_iPadlen/2) * (m_iFrameHeight/2 + 2*m_iPadlen/2);}
	Int getLStride(){return m_iFrameWidth + 2*m_iPadlen;}
	Int getCStride(){return m_iFrameWidth/2 + 2*m_iPadlen/2;}
	Int getLOrgOffset(){return m_iPadlen   * getLStride() + m_iPadlen;}
	Int getCOrgOffset(){return m_iPadlen/2 * getCStride() + m_iPadlen/2;}

	Int m_iFrameWidth;
	Int m_iFrameHeight;
	Int m_iPadlen;
  	Int m_iGOPSize;

// 	Store all frames
	std::vector<Pel*> m_cYList;
	std::vector<Pel*> m_cUList;
	std::vector<Pel*> m_cVList;

	Pel*   m_pYOrgPU;
	Pel*   m_pUOrgPU;
	Pel*   m_pVOrgPU;
//  reference frame parameters
	Int m_iRefId;
	Int m_iRefListId;

//  reference frame lists
	std::vector<Pel*> m_cYRefList0;
	std::vector<Pel*> m_cURefList0;
	std::vector<Pel*> m_cVRefList0;

	std::vector<Pel*> m_cYRefList1;
	std::vector<Pel*> m_cURefList1;
	std::vector<Pel*> m_cVRefList1;

	Pel*   m_pYRefPU;
	Pel*   m_pURefPU;
	Pel*   m_pVRefPU;

//  Cost structure
	Int m_iMvPredx;
	Int m_iMvPredy;
	Int m_iCostScale;
	UInt m_uiCost;
	UInt* m_puiSadArray;
	UInt* m_puiSadArrayCPU;
	
// PU parameters
	Int m_iLConstrain;
	Int m_iRConstrain;
	Int m_iTConstrain;
	Int m_iBConstrain;
	Int m_iPUId;

	Int m_ilx;
	Int m_irx;
	Int m_ity;
	Int m_iby;
	
	Int m_iPUWidth;
	Int m_iPUHeight;
	Int m_iPUOffset;

//Search parameters
	Int m_iSearchRange;
  	Int m_biPredSearchRange; // Search range for bi-prediction
  	Int m_iFastSearch;

//Result
  	Int m_iMvX;
  	Int m_iMvY;
  	UInt m_uiSad;
	

};

class GpuMeDataAccess:public CpuMeDataAccess
{
public:
	GpuMeDataAccess(Int iFrameWidth = 640, Int iFrameHeight = 480, Int iGOPSize = 8): 
					CpuMeDataAccess(iFrameWidth, iFrameHeight, iGOPSize)
	{}
	
	~GpuMeDataAccess()
	{}
	
	void init();
	void destroy();
	//void initSearchData(Int iSearchRange = 128, Int iBiPredSearchRange = 16, Int iFastSearch = 0);
};



//Incorporate into TEncSearch class
class GpuSearch
{
public:
	GpuSearch(Int iSearchRange = 128, Int iBiPredSearchRange = 16, Int iFastSearch = 0)
	{
		m_iSearchRange = iSearchRange;
		m_biPredSearchRange = iBiPredSearchRange;
		m_iFastSearch = iFastSearch;
	}
	
	~GpuSearch(){}
	
	void init();
	void xMotionEstimation(CpuMeDataAccess* pCpuME, CpuMeDataAccess* pGpuME = NULL);
	void xFullBlockSearch(CpuMeDataAccess* pGpuME);
	void xBlockFastSearch(CpuMeDataAccess* pCpuME);
	void xHeteroSearch(CpuMeDataAccess* pCpuME, CpuMeDataAccess* pGpuME = NULL);

	Int m_iSearchRange;
  	Int m_biPredSearchRange; // Search range for bi-prediction
  	Int m_iFastSearch;
};

#endif

