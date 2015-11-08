#! /usr/bin/

alg=TZsearch
alg2=GpuSearch
DataFolder=ScalingFastOneSecond
for fname in {"BlowingBubbles","RaceHorses","Cactus","PeopleOnStreet"} 
do
	#echo Encoding ${fname}${alg}
	#./TAppEncoderStaticSF1 -c ../cfg/groupcfg/${alg}/${fname}.cfg > ${DataFolder}/${fname}${alg}QP27
	#echo Encoding ${fname}${alg2}SF1
	#./TAppEncoderStaticSF1 -c ../cfg/groupcfg/${alg2}/${fname}.cfg > ${DataFolder}/${fname}${alg2}SF1QP27
	#echo Encoding ${fname}${alg2}SF2
	#./TAppEncoderStaticSF2 -c ../cfg/groupcfg/${alg2}/${fname}.cfg > ${DataFolder}/${fname}${alg2}SF2QP27
	#echo Encoding ${fname}${alg2}SF4
	#./TAppEncoderStaticSF4 -c ../cfg/groupcfg/${alg2}/${fname}.cfg > ${DataFolder}/${fname}${alg2}SF4QP27
	#echo Encoding ${fname}${alg2}SF8
	#./TAppEncoderStaticSF8 -c ../cfg/groupcfg/${alg2}/${fname}.cfg > ${DataFolder}/${fname}${alg2}SF8QP27
	
	
	echo Encoding ${fname}${alg2}SF4
	./TAppEncoderStaticSF4 -c ../cfg/groupcfg/${alg2}/${fname}.cfg > ./${fname}${alg2}SF4QP27
done

