

declare -a var
init_count=$count 
while read p; do
      echo $p
      #script="source /usr/share/Modules/3.2.10/init/bash && CUDA_VISIBLE_DEVICES=0 nohup ${exp_dir}/exp_${count}.sh > ${exp_dir}/LOG_${count} 2>&1 &"
      script="CUDA_VISIBLE_DEVICES=0 nohup ${exp_dir}/exp_${count}.sh > ${exp_dir}/LOG_${count} 2>&1 &"
      echo $script
      ssh -o StrictHostKeyChecking=no -n -f ${USER}@$p $script
      var[`expr $count - $init_count`]=${exp_dir}/JACK_$count  
      count=`expr $count + 1`

      #script="source /usr/share/Modules/3.2.10/init/bash && CUDA_VISIBLE_DEVICES=1 nohup ${exp_dir}/exp_${count}.sh > ${exp_dir}/LOG_${count} 2>&1 &"
      script="CUDA_VISIBLE_DEVICES=1 nohup ${exp_dir}/exp_${count}.sh > ${exp_dir}/LOG_${count} 2>&1 &"
      echo $script
      ssh -o StrictHostKeyChecking=no -n -f ${USER}@$p $script  
      var[`expr $count - $init_count`]=${exp_dir}/JACK_$count  
      count=`expr $count + 1`
  
done <$PBS_NODEFILE

for i in "${var[@]}" 
do 
	echo $i 
    until [ -f $i ]
    do
        sleep 10
    done

done 


