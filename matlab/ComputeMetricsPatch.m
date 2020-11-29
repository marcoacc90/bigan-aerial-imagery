function [p,n,tp,tn,fp,fn,acc, precision, sensitivity, specificity,fscore, mcc,th]  = ComputeMetricsPatch( normal, novel, n_thresholds )

data = [normal(:);novel(:)];
%[p, npatch_normal] = size( normal )
%[n, npatch_novel ] = size( novel )

normalData = normal(:);
novelData = novel(:);

p = length(normalData);
n = length(novelData);

th = linspace( min( data ), max( data ), n_thresholds );
tp = zeros( 1,length( th ) );
tn = zeros( 1,length( th ) );
fp = zeros( 1,length( th ) );
fn = zeros( 1,length( th ) );
acc = zeros( 1,length( th ) );
precision = zeros( 1,length( th ) );
sensitivity = zeros( 1,length( th ) );
specificity = zeros( 1,length( th ) );
fscore =  zeros( 1,length( th ) );
mcc = zeros( 1,length( th ) );

for i = 1 : length( th )    
    tp(i) = length(find( normalData <= th(i) ));
    fn(i) = p - tp(i);
    tn(i) = length(find( novelData > th(i) )); 
    fp(i) = n - tn(i);  

    acc(i) = ( tp(i) + tn(i) ) / (p + n);
    precision(i) = tp(i) /(tp(i) + fp(i));
    sensitivity(i) = tp(i) / p;
    specificity(i) = tn(i) / n;
    fscore(i) = ( 2 * tp(i)) / (2*tp(i) + fp(i) + fn(i));
     
    den = sqrt( (tp(i) + fp(i)) * (tp(i) + fn(i)) * (tn(i) + fp(i)) * (tn(i) + fn(i) ) );
    if den < 1e-12 
        den = 1;
    end
    mcc(i) = ( tp(i)*tn(i) - fp(i)*fn(i) )/den;
end

