docker run --cap-add SYS_ADMIN -itd --device /dev/bmdev-ctl --device /dev/bm-sophon5:/dev/bm-sophon0 -v /home/aigc/rzy_backup/realtimeASR:/workspace -v /home/aigc/rzy_backup/libsophon-0.5.1:/opt/sophon/libsophon-current -w /workspace --name chattts sophgo/tpuc_dev:latest bash