wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz
tar -xzf openscene_metadata_test.tgz
rm openscene_metadata_test.tgz

printf "%s\n" {0..31} | xargs -I {} -P 8 bash -c "
    echo 'Processing camera test split {}...'
    wget -qO- https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_{}.tgz | tar -xz
"

# printf "%s\n" {0..31} | xargs -I {} -P 8 bash -c "
#     echo 'Processing lidar test split {}...'
#     wget -qO- https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_lidar/openscene_sensor_test_lidar_{}.tgz | tar -xz
# "

mv openscene-v1.1/meta_datas test_navsim_logs
mv openscene-v1.1/sensor_blobs test_sensor_blobs
rm -r openscene-v1.1
