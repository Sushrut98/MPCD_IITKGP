close all
clc

x = linspace(x_min,x_max,ncx_fp);
y = linspace(y_min,y_max,ncy_fp);

disect = 31;
u_plot = u_f_run(:,:,disect);
v_plot = v_f_run(:,:,disect);
w_plot = w_f_run(:,:,disect);

[X,Y] = meshgrid(x,y);

velocity = sqrt(u_plot.^2+v_plot.^2+w_plot.^2);
Rad = 4;

contourf(X,Y,velocity)
hold on

% startH = linspace(-pi/2,pi/2,50);
% startr = Rad*ones(length(startH),1);
% starty = startr.*sin(startH);
% startx = startr.*cos(startH);
% startH1 = linspace(pi/2,3*pi/2,50);
% startr1 = Rad*ones(length(startH1),1);
% starty1 = startr1.*sin(startH1);
% startx1 = startr1.*cos(startH1);
% streamline(X,Y,U,V,startx,starty)
% hold on
% streamline(X,Y,U,V,startx1,starty1)

% daspect([1,1,1])
% hold on

H = linspace(0,2*pi,50);

x = Rad*cos(H);
y = Rad*sin(H);

plot(x,y)
daspect([1,1,1])
hold on

U = u_plot;
V = v_plot;

quiver(X,Y,U,V)
daspect([1,1,1])
hold on
%% Plotting Style 1 - crossing lines
% startx = linspace(-15,15,ncx_fp);
% starty = linspace(-15,15,ncy_fp);
% streamline(X,Y,u_plot,v_plot,startx,starty)

%% Plotting Style 2 - parallel lines to axis
% startx = -10:0.5:10;
% starty = 0*ones(1,length(startx));
% streamline(X,Y,U,V,startx,starty)
% hold on

% figure(2)
% H = linspace(0,2*pi,50);
% 
% x = Rad*cos(H);
% y = Rad*sin(H);
% 
% plot(x,y)
% daspect([1,1,1])
% hold on
% starty = -10:0.25:10;
% startx = 15*ones(1,length(starty));
% streamline(X,Y,U,V,startx,starty)
% hold on
% %% Plotting Style 3 - around sphere
% startH = linspace(-pi/2,pi/2,50);
% startr = Rad*ones(length(startH),1);
% starty = startr.*sin(startH);
% startx = startr.*cos(startH);
% startH1 = linspace(pi/2,3*pi/2,50);
% startr1 = Rad*ones(length(startH1),1);
% starty1 = startr1.*sin(startH1);
% startx1 = startr1.*cos(startH1);
% streamline(X,Y,U,V,startx,starty)
% hold on
% streamline(X,Y,U,V,startx1,starty1)