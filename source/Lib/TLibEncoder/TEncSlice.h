/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2015, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     TEncSlice.h
    \brief    slice encoder class (header)
*/

#ifndef __TENCSLICE__
#define __TENCSLICE__

// Include files
#include "TLibCommon/CommonDef.h"
#include "TLibCommon/TComList.h"
#include "TLibCommon/TComPic.h"
#include "TLibCommon/TComPicYuv.h"
#include "TEncCu.h"
#include "WeightPredAnalysis.h"
#include "TEncRateCtrl.h"
#include <thread>
#include <condition_variable>
#include <mutex>
#include <pthread.h>

//! \ingroup TLibEncoder
//! \{

class TEncTop;
class TEncGOP;

// ====================================================================================================================
// Class definition
// ====================================================================================================================

/// slice encoder class

class WPPScheduler{
public:
  WPPScheduler():m_uiCtuInRow(0),m_uiCtuInColumn(0),m_pWppOrderIndex(NULL)
  {

  }

  WPPScheduler(int uiCtuInRow, int uiCtuInColumn):
    m_uiCtuInRow(uiCtuInRow),m_uiCtuInColumn(uiCtuInColumn)
  {
    m_pWppOrderIndex = new int[uiCtuInRow * uiCtuInColumn];
    m_ctuStatus      = new int[uiCtuInRow * uiCtuInColumn];
  }

  void SetWPPScheduler(int uiCtuInRow, int uiCtuInColumn)
  {
    m_uiCtuInRow = uiCtuInRow;
    m_uiCtuInColumn = uiCtuInColumn;

    if(m_pWppOrderIndex==NULL)
      m_pWppOrderIndex = new int[uiCtuInRow * uiCtuInColumn];

    if(m_ctuStatus==NULL)
      m_ctuStatus      = new int[uiCtuInRow * uiCtuInColumn];

  }

  ~WPPScheduler(){
    if(m_pWppOrderIndex!=NULL)
      delete m_pWppOrderIndex;

    if(m_ctuStatus!=NULL)
      delete m_ctuStatus;
  }

  void CalculateWppCtuAddress()
  {
    int uiCtuWFAddress=0;
    int uiDiagStartX=0;
    int uiDiagStartY=0;
    int uiCtuInRow = m_uiCtuInRow;
    int uiCtuInColumn = m_uiCtuInColumn;
    int uiCtuAddress;
    int uiCtuY,uiCtuX;
    int TotalCtu = uiCtuInRow * uiCtuInColumn;
      
    for(uiCtuAddress=0;uiCtuAddress<TotalCtu;uiCtuAddress++)
    {
      uiCtuY           = uiCtuWFAddress / uiCtuInRow;
      uiCtuX          = uiCtuWFAddress % uiCtuInRow;

      m_pWppOrderIndex[uiCtuAddress] = uiCtuWFAddress;
        
      if(uiCtuX != 0 && uiCtuY != (uiCtuInColumn -1) && uiDiagStartX != (uiCtuInRow - 1)) // 30 is a magic nuCtuer to be replaced by uiCtuInColumn
      {
        if( (uiCtuWFAddress + uiCtuInRow - 2) < (uiCtuWFAddress/uiCtuInRow + 1) * uiCtuInRow)
              {
                  uiDiagStartX++;
                  uiCtuWFAddress = uiDiagStartX + uiDiagStartY * uiCtuInRow;
              } else {
                  uiCtuWFAddress = uiCtuWFAddress + (uiCtuInRow - 2);
            }
      } else {
              if( uiDiagStartX > (uiCtuInRow - 1) || uiDiagStartY > (uiCtuInColumn - 1))
                  printf("Wave front overflow!");
              
        if ( uiDiagStartX != uiCtuInRow - 1) {
                  uiDiagStartX += 1;
 
              } else {
                  uiDiagStartX -= 2;
                  uiDiagStartY += 1;
              }
         
        uiCtuWFAddress = uiDiagStartX + uiDiagStartY * uiCtuInRow;
      }
        }
    
    m_pWppOrderIndex[uiCtuWFAddress] = uiCtuWFAddress;
  }

  void printWppIndex() const {
    printf("\n");
    for(int i=0;i<m_uiCtuInRow*m_uiCtuInColumn;i++)
      printf("%d,",m_pWppOrderIndex[i]);
    printf("\n");

  }

  int getWppIndex(int RasterScanOrderIndex) const {
    return m_pWppOrderIndex[RasterScanOrderIndex];
  }

  int getCtuInRow() const {
    return m_uiCtuInRow;
  }

  int getCtuInColumn() const {
    return m_uiCtuInColumn;
  }
  
  void resetStatus()
  {
    memset(m_ctuStatus,0,sizeof(int)*m_uiCtuInRow*m_uiCtuInColumn);
  }

public:
  int *m_ctuStatus;

private:
  int m_uiCtuInRow;
  int m_uiCtuInColumn;
  int *m_pWppOrderIndex;
};

class TEncSlice
  : public WeightPredAnalysis
{
    
    //multi-threading
public:
 
    typedef struct threadpool {
        // you should fill in this structure with whatever you need
        int ctuIndex;
        int num_threads;	//number of active threads
        std::thread *threads;	//pointer to threads
        std::mutex qlock;		//lock on the queue list
        std::mutex cs;
        std::condition_variable q_not_empty;	//non empty and empty condidtion vairiables
        int shutdown;
        int dont_accept;
    } threadpool;
    
    threadpool m_threadpool;
    WPPScheduler m_cWppScheduler;
private:
  // encoder configuration
  TEncCfg*                m_pcCfg;                              ///< encoder configuration class

  // pictures
  TComList<TComPic*>*     m_pcListPic;                          ///< list of pictures
  TComPicYuv*             m_apcPicYuvPred;                      ///< prediction picture buffer
  TComPicYuv*             m_apcPicYuvResi;                      ///< residual picture buffer

  // processing units
  TEncGOP*                m_pcGOPEncoder;                       ///< GOP encoder
  TEncCu*                 m_pcCuEncoder;                        ///< CU encoder
  TEncCu*                 m_pcCuEncoderWPP;
  // encoder search
  TEncSearch*             m_pcPredSearch;                       ///< encoder search class

  // coding tools
  TEncEntropy*            m_pcEntropyCoder;                     ///< entropy encoder
  TEncSbac*               m_pcSbacCoder;                        ///< SBAC encoder
  TEncBinCABAC*           m_pcBinCABAC;                         ///< Bin encoder CABAC
  TComTrQuant*            m_pcTrQuant;                          ///< transform & quantization

  // RD optimization
  TComRdCost*             m_pcRdCost;                           ///< RD cost computation
  TEncSbac***             m_pppcRDSbacCoder;                    ///< storage for SBAC-based RD optimization
  TEncSbac*               m_pcRDGoOnSbacCoder;                  ///< go-on SBAC encoder
  UInt64                  m_uiPicTotalBits;                     ///< total bits for the picture
  UInt64                  m_uiPicDist;                          ///< total distortion for the picture
  Double                  m_dPicRdCost;                         ///< picture-level RD cost
  Double*                 m_pdRdPicLambda;                      ///< array of lambda candidates
  Double*                 m_pdRdPicQp;                          ///< array of picture QP candidates (double-type for lambda)
  Int*                    m_piRdPicQp;                          ///< array of picture QP candidates (Int-type)
  TEncRateCtrl*           m_pcRateCtrl;                         ///< Rate control manager
  UInt                    m_uiSliceIdx;
  TEncSbac                m_lastSliceSegmentEndContextState;    ///< context storage for state at the end of the previous slice-segment (used for dependent slices only).
  TEncSbac                m_entropyCodingSyncContextState;      ///< context storate for state of contexts at the wavefront/WPP/entropy-coding-sync second CTU of tile-row
  SliceType               m_encCABACTableIdx;

  Void     setUpLambda(TComSlice* slice, const Double dLambda, Int iQP);
  Void     calculateBoundingCtuTsAddrForSlice(UInt &startCtuTSAddrSlice, UInt &boundingCtuTSAddrSlice, Bool &haveReachedTileBoundary, TComPic* pcPic, const Int sliceMode, const Int sliceArgument, const UInt uiSliceCurEndCtuTSAddr);

public:
  TEncSlice();
  virtual ~TEncSlice();
   
  void encodeCTUWPP(TComPic* pcPic, UInt ctuTsAddr, TComSlice* const pcSlice, const UInt frameWidthInCtus, UInt startCtuTsAddr,  UInt boundingCtuTsAddr, TEncBinCABAC* pRDSbacCoder, TComBitCounter* tempBitCounter, Int ctu_row_id, Int ThreadId);
  void threadTask(TComPic* pcPic, UInt ctuTsAddr, TComSlice* const pcSlice, const UInt frameWidthInCtus, UInt startCtuTsAddr,  UInt boundingCtuTsAddr, TEncBinCABAC* pRDSbacCoder, TComBitCounter* tempBitCounter, Int threadid);
    
  Void    create              ( Int iWidth, Int iHeight, ChromaFormat chromaFormat, UInt iMaxCUWidth, UInt iMaxCUHeight, UChar uhTotalDepth );
  Void    destroy             ();
  Void    init                ( TEncTop* pcEncTop );

  /// preparation of slice encoding (reference marking, QP and lambda)
  Void    initEncSlice        ( TComPic*  pcPic, Int pocLast, Int pocCurr, Int iNumPicRcvd,
                                Int iGOPid,   TComSlice*& rpcSlice, Bool isField );
  Void    resetQP             ( TComPic* pic, Int sliceQP, Double lambda );
  // compress and encode slice
  Void    precompressSlice    ( TComPic* pcPic                                     );      ///< precompress slice for multi-loop opt.
  Void    compressSlice       ( TComPic* pcPic                                     );      ///< analysis stage of slice
  Void    calCostSliceI       ( TComPic* pcPic );
  Void    encodeSlice         ( TComPic* pcPic, TComOutputBitstream* pcSubstreams, UInt &numBinsCoded );

  // misc. functions
  Void    setSearchRange      ( TComSlice* pcSlice  );                                  ///< set ME range adaptively

  TEncCu*        getCUEncoder() { return m_pcCuEncoder; }                        ///< CU encoder
  Void    xDetermineStartAndBoundingCtuTsAddr  ( UInt& startCtuTsAddr, UInt& boundingCtuTsAddr, TComPic* pcPic );
  UInt    getSliceIdx()         { return m_uiSliceIdx;                    }
  Void    setSliceIdx(UInt i)   { m_uiSliceIdx = i;                       }

  SliceType getEncCABACTableIdx() const           { return m_encCABACTableIdx;        }

private:
  Double  xGetQPValueAccordingToLambda ( Double lambda );
};

//! \}

#endif // __TENCSLICE__
