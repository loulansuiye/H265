#! /usr/bin/

alg=BaseTime


for fname in {"BasketBallDrillText","BasketBallDrive","BasketBallPass","BQTerrace","Johny","Kimono1","KristenAndSara","ParkScene","SlideEditing"} 
do
	echo Encoding ${fname}_QP22.cfg
	./TAppEncoderStatic -c ../cfg/groupcfg/${alg}/${fname}_QP22.cfg > ${fname}${alg}SF4QP22
	echo Encoding ${fname}_QP27.cfg
	./TAppEncoderStatic -c ../cfg/groupcfg/${alg}/${fname}_QP27.cfg > ${fname}${alg}SF4QP27
	echo Encoding ${fname}_QP32.cfg
	./TAppEncoderStatic -c ../cfg/groupcfg/${alg}/${fname}_QP32.cfg > ${fname}${alg}SF4QP32
	echo Encoding ${fname}_QP37.cfg
	./TAppEncoderStatic -c ../cfg/groupcfg/${alg}/${fname}_QP37.cfg > ${fname}${alg}SF4QP37
done

