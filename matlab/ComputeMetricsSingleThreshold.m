function [p,n,tp,tn,fp,fn,acc, precision, sensitivity, specificity,fscore, mcc]  = ComputeMetricsSingleThreshold( normal, novel, th )

p = length(normal);
n = length(novel);
    
tp = length(find( normal <= th ));
fn = p - tp;
tn = length(find( novel > th )); 
fp = n - tn;  

acc = ( tp + tn ) / (p + n);
precision = tp /(tp + fp);
sensitivity = tp / p;
specificity = tn / n;
fscore = ( 2 * tp ) / (2*tp + fp + fn );
     
den = sqrt( (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn ) );
if den < 1e-12 
    den = 1;
end
mcc = ( tp*tn - fp*fn )/den;

