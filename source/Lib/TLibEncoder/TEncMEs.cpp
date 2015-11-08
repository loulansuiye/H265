#include "TEncMEs.h"
#include <stdlib.h>
///////////////////////////Class CpuMeDataAccess Starts/////////////////////////////////

void 
CpuMeDataAccess::init()
{
	//May not be necessary since HM has its own list for storing frames.

	//Pel* pTmp;
	
	// for(Int i=0; i < m_iGOPSize+1; i++)
	// {
	// 	pTmp = new Pel[getLPadSize()];
	// 	m_cYList.push_back(pTmp);

	// 	pTmp = new Pel[getCPadSize()];
	// 	m_cUList.push_back(pTmp);

	// 	pTmp = new Pel[getCPadSize()];
	// 	m_cVList.push_back(pTmp);

	// }
	
	// for(Int i=0; i < m_iGOPSize+1; i++)
	// {
	// 	pTmp = new Pel[getLPadSize()];
	// 	m_cYRefList.push_back(pTmp);
		
	// 	pTmp = new Pel[getCPadSize()];
	// 	m_cURefList.push_back(pTmp);
		
	// 	pTmp = new Pel[getCPadSize()];
	// 	m_cVRefList.push_back(pTmp);
	// }
}

void 
CpuMeDataAccess::destroy()
{
	// for(Int i=0; i < m_iGOPSize+1; i++)
	// {
	// 	delete m_cYList[i];		
	// 	delete m_cUList[i];
	// 	delete m_cVList[i];
	// 	delete m_cYRefList[i];
	// 	delete m_cURefList[i];
	// 	delete m_cVRefList[i];
	// }
}
///////////////////////////Class CpuMeDataAccess Ends/////////////////////////////////





///////////////////////////Class GpuMeDataAccess Starts/////////////////////////////////
void 
GpuMeDataAccess::init()
{
	Pel* pTmp;

	for(Int i=0; i < m_iGOPSize+1; i++)
	{
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * getLPadSize());
		m_cYList.push_back(pTmp);

		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * getCPadSize());
		m_cUList.push_back(pTmp);

		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * getCPadSize());
		m_cVList.push_back(pTmp);

	}
	
	for(Int i=0; i < m_iGOPSize+1; i++)
	{
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * getLPadSize());
		m_cYRefList0.push_back(pTmp);
		
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * getCPadSize());
		m_cURefList0.push_back(pTmp);
		
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * getCPadSize());
		m_cVRefList0.push_back(pTmp);
	}

	for(Int i=0; i < m_iGOPSize+1; i++)
	{
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * getLPadSize());
		m_cYRefList1.push_back(pTmp);
		
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * getCPadSize());
		m_cURefList1.push_back(pTmp);
		
		GPU::MemOpt::AllocDeviceMem<Pel*>(pTmp,sizeof(Pel) * getCPadSize());
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


//GpuMeDataAccess will be initialized 
//many times because of duplicating cSearch
//However, they shared current frame and reference lists 
//The way to solve is initialize HM own cSearch first
//Then for each cSearch after, it makes aliases to the
//first cSearch allocated GPU memory.

void 
GpuMeDataAccess::initWPP(GpuMeDataAccess* m_pcHostGPU)
{

	//Alias current frame
	if(m_pcHostGPU->m_cYList.size()!=0)
	{
		m_cYList = m_pcHostGPU->m_cYList;
	} else {
		printf("Alias m_cYList error!\n");
	}

	if(m_pcHostGPU-m_cUList.size()!=0)
	{
		m_cUList = m_pcHostGPU->m_cUList;
	} else {
		printf("Alias m_cUList error!\n");
	}

	if(m_pcHostGPU->m_cVList.size()!=0)
	{
		m_cVList = m_pcHostGPU->m_cVList;
	} else {
		printf("Alias m_cUList error!\n");
	}

	//Alias reference list 0
	if(m_pcHostGPU->m_cYRefList0.size()!=0)
	{
		m_cYRefList0 = m_pcHostGPU->m_cYRefList0;
	} else {
		printf("Alias m_cYRefList0 error!\n");
	}

	if(m_pcHostGPU->m_cURefList0.size()!=0)
	{
		m_cURefList0 = m_pcHostGPU->m_cURefList0;
	} else {
		printf("Alias m_cURefList0 error!\n");
	}

	if(m_pcHostGPU->m_cVRefList0.size()!=0)
	{
		m_cVRefList0 = m_pcHostGPU->m_cVRefList0;
	} else {
		printf("Alias m_cVRefList0 error!\n");
	}

	//Alias reference list 1
	if(m_pcHostGPU->m_cYRefList1.size()!=0)
	{
		m_cYRefList1 = m_pcHostGPU->m_cYRefList1;
	} else {
		printf("Alias m_cYRefList1 error!\n");
	}

	if(m_pcHostGPU->m_cURefList1.size()!=0)
	{
		m_cURefList1 = m_pcHostGPU->m_cURefList1;
	} else {
		printf("Alias m_cURefList1 error!\n");
	}

	if(m_pcHostGPU->m_cVRefList1.size()!=0)
	{
		m_cVRefList1 = m_pcHostGPU->m_cVRefList1;
	} else {
		printf("Alias m_cVRefList1 error!\n");
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









void 
GpuMeDataAccess::destroy()
{
	for(Int i=0; i < m_iGOPSize+1; i++)
	{
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

///////////////////////////Class GpuMeDataAccess Ends/////////////////////////////////


///////////////////////////Class GpuSearch Starts/////////////////////////////////


void
GpuSearch::init()
{

}

void
GpuSearch::xMotionEstimation(CpuMeDataAccess* pCpuME, CpuMeDataAccess* pGpuME)
{

}

void
GpuSearch::xFullBlockSearch(CpuMeDataAccess* pGpuME)
{

}

void
GpuSearch::xBlockFastSearch(CpuMeDataAccess* pCpuME)
{

}

void 
xHeteroSearch(CpuMeDataAccess* pCpuME, CpuMeDataAccess* pGpuME = NULL)
{
	
}

///////////////////////////Class GpuSearch Ends/////////////////////////////////
