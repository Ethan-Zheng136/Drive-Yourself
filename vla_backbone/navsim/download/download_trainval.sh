wget -qO- https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz | tar -xz

printf "%s\n" {0..199} | xargs -I {} -P 8 bash -c "
    echo 'Processing camera split {}...'
    wget -qO- https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_{}.tgz | tar -xz
"

# printf "%s\n" {0..199} | xargs -I {} -P 8 bash -c "
#     echo 'Processing lidar trainval split {}...'
#     wget -qO- https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_lidar/openscene_sensor_trainval_lidar_{}.tgz | tar -xz
# "

mv openscene-v1.1/meta_datas trainval_navsim_logs
mv openscene-v1.1/sensor_blobs trainval_sensor_blobs
rm -r openscene-v1.1