close all
clear all;
clc

%%%% SELECT
EPOCHS = 500;
MODEL = 'IZI';
th = 0.055;


% DO NOT CHANGE
mode = 'Test';


path = sprintf('./../E%d_Results', EPOCHS );
oname = sprintf('%s/%s_metrics_%s_patch.txt',path,MODEL,mode);
fileID = fopen( oname, 'w' );
fprintf(fileID,'\n%s\n',mode);
name = sprintf('%s/%s_loss_anomaly_%s.txt',path,MODEL,mode);
novel = load(name);
name = sprintf('%s/%s_loss_normal_%s.txt',path,MODEL,mode);
normal = load(name);

novel = novel(:);
normal = normal(:);


[p,n,tp,tn,fp,fn,acc, precision, sensitivity, specificity,fscore,mcc] = ComputeMetricsSingleThreshold( normal, novel, th ); 
fprintf(fileID,'ACC         = %f\n', acc );
fprintf(fileID,'Precison    = %f\n', precision );
fprintf(fileID,'Sensitivity = %f\n', sensitivity );
fprintf(fileID,'Specificity = %f\n', specificity );
fprintf(fileID,'Fscore      = %f\n', fscore );
fprintf(fileID,'MCC         = %f\n', mcc );
 
fclose(fileID);
cmd = sprintf('%s is ready!!!',oname);
disp(cmd)

