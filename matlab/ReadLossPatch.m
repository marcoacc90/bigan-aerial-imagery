clear all
clc
close all

%%%% SELECT
EPOCHS = 500;
MODEL = 'IZI'; % IZIf, IZI, ZIZ
mode = 'Test';  % Test, Validation
bin = 50;

path = sprintf('./../E%d_Results', EPOCHS );
name = sprintf('%s/%s_loss_anomaly_%s.txt',path,MODEL,mode);
novel = load(name);
name = sprintf('%s/%s_loss_normal_%s.txt',path,MODEL,mode);
normal = load(name);

histogram(normal(:),bin)
hold on
histogram(novel(:),bin)

%title(mode)
xlabel('Anomaly Score')
ylabel('Patches')
grid on
legend('Normal','Anomaly')
set(gca,'FontSize',18)
name = sprintf('%s/%s_hist_%s_patch.png', path,MODEL,mode );
saveas(gcf,name)
