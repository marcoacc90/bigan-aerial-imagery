clear all
clc
close all

%%%% SELECT
EPOCHS = 210;
MODEL = 'E1000_BIGAN';
mode = 'Test';
bin = 50;
metric = 2; %1,2

path = sprintf('./../E%d_Results', EPOCHS );
name = sprintf('%s/%s_loss_anomaly_%s.txt',path,MODEL,mode);
novel = load(name);
name = sprintf('%s/%s_loss_normal_%s.txt',path,MODEL,mode);
normal = load(name);

histogram(normal(:,metric),bin)
hold on
histogram(novel(:,metric),bin)

%title(mode)
xlabel('Anomaly Score')
ylabel('Patches')
grid on
legend('Normal','Anomaly')
set(gca,'FontSize',18)
name = sprintf('%s/%s_hist_%s_patch.png', path,MODEL,mode );
saveas(gcf,name)
