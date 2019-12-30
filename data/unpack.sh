#!/usr/bin/env bash
cat ad_data_p.tar.gz.part* > ad_data.tar.gz
tar -vxf ad_data.tar.gz
cd pose
./unpack_pose.sh
