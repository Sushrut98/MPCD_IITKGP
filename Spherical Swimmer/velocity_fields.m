clc
clear all
close all

R = 1;
beta = -5;

x=linspace(-20,20,1000);
y=linspace(-20,20,1000);
[X,Y]=meshgrid(x,y);

r = sqrt(X.^2 + Y.^2);
costheta = X./r;
sintheta = Y./r;

%%First Term Contribution
u = (1./r.^3).*(costheta.^2 - 1/3);
v = (1./r.^3).*(costheta.*sintheta);

%%Second Term Contribution

u = u + beta*(1./r.^4 - 1./r.^2).*0.5*(-1+3.*costheta.^2).*costheta;
v = v + beta*(1./r.^4 - 1./r.^2).*0.5*(-1+3.*costheta.^2).*sintheta;

%Third Term Contribution

u = u + beta*(1./r.^4).*costheta.*(costheta.^2-1);
v = v + beta*(1./r.^4).*costheta.*(costheta.*sintheta);

insq = find(X.^2+Y.^2 < R^2);
u(insq) = 0;
v(insq) = 0;
% starty = -10:0.5:10;
% startx = 5*ones(length(starty),1);%-10:0.5:10;

startH = linspace(-pi/2,pi/2,50);
startr = 1*ones(length(startH),1);

starty = [startr.*sin(startH)];
startx = [startr.*cos(startH)];


quiver(X,Y,u,v)
daspect([1,1,1])
hold on

streamline(X,Y,u,v,startx,starty)
hold on

startH1 = linspace(pi/2,3*pi/2,50);
startr1 = 1*ones(length(startH1),1);

starty1 = [startr1.*sin(startH1)];
startx1 = [startr1.*cos(startH1)];

Velocity = sqrt(u.^2 + v.^2);
contourf(X,Y,Velocity)
hold on

% streamline(X,Y,u,v,startx1,starty1)
% hold on
countourx=cos(0:pi/100:2*pi);
countoury=sin(0:pi/100:2*pi);
plot(countourx,countoury)
daspect([1,1,1])