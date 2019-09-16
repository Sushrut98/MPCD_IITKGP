clear all
close all


Niter = 1;
ncx = 51;
ncy = 51;
ncz = 51;
nc_total = (ncx)*(ncy)*(ncz);

h = 1;
h2 = h/2;

Lx = (ncx-1)*h;              
Ly = (ncy-1)*h;
Lz = (ncz-1)*h;

center_b_x = 0;
center_b_y = 0;
center_b_z = 0;

x_min = center_b_x - Lx/2;
y_min = center_b_y - Ly/2;
z_min = center_b_z - Lz/2;

x_max = x_min + Lx;
y_max = y_min + Ly;
z_max = z_min + Lz;

% np_avg = 10;
% N = (ncx-1)*(ncy-1)*(ncz-1)*np_avg;
N=1;
R1 = 4*h;
% R2 = R1 - 1.8*h
% N_v = (4/3)*pi*(R1**3-R2**3)*np_avg
% N_v=int(N_v)
mass = 1;

% res = 2;
% hcf = h/res;
% ncx_fp = res*(ncx-1)
% ncy_fp = res*(ncy-1)
% ncz_fp = res*(ncz-1)
% nct_profile = ncx_fp*ncy_fp*ncz_fp

mu = 0;
kBT = 36;
sigma = sqrt(kBT/mass);
lamda = 0.1;

beta = 0;
% B1 = 0.01*sqrt(kBT/mass);
B1=1;     
%dt = lamda/sigma;
dt=1;
dt2 = dt/2;

%% Initialize position and velocity

x=4.46
y=2
z=0
u=-1.5
v=0
w=0
xmin=-6;
xmax=6;
ymin=-6;
ymax=6;
zmin=-6;
zmax=6;
%% Plotting Sphere
[a b c]=sphere;
figure(1);
surf(R1*a,R1*b,R1*c,'FaceAlpha',0.5 ,'EdgeColor', 'none')
hold on

figure(1)
quiver3(x,y,z,u,v,w,'filled', 'Marker', 'o', 'LineWidth', 1.8,'AutoScaleFactor', 0.5)
daspect([1 1 1])
axis([xmin xmax ymin ymax zmin zmax])
hold on
pause()




%% Simulation begins 
 
%% Streamimg Step

x=x+u*dt
y=y+v*dt
z=z+w*dt
%disp(i)

figure(1)
quiver3(x,y,z,u,v,w,'filled', 'Marker', 'o', 'LineWidth', 1.8,'AutoScaleFactor', 0.5)
daspect([1 1 1])
axis([xmin xmax ymin ymax zmin zmax])
hold on
pause()
%% Pereodic Boundary Condition
 
x=x_min+mod((x-x_min),Lx)
y=y_min+mod((y-y_min),Ly)
z=z_min+mod((z-z_min),Lz)

figure(1)
quiver3(x,y,z,u,v,w,'filled', 'Marker', 'o', 'LineWidth', 1.8,'AutoScaleFactor',0.5)
daspect([1 1 1])
axis([xmin xmax ymin ymax zmin zmax])
hold on
pause()
%% Fluid-squirmer Interaction Boundary Condition

sq_in=[];
sq_in = find((x.^2+y.^2+z.^2)<=(R1^2));
    
x(sq_in) = x(sq_in) - u(sq_in)*dt2
y(sq_in) = y(sq_in) - v(sq_in)*dt2
z(sq_in) = z(sq_in) - w(sq_in)*dt2

figure(1)
quiver3(x,y,z,u,v,w,'filled', 'Marker', 'o', 'LineWidth', 1.8,'AutoScaleFactor',0.5)
daspect([1 1 1])
axis([xmin xmax ymin ymax zmin zmax])
hold on
pause()


mag_position = sqrt(x(sq_in)^2 + y(sq_in)^2 + z(sq_in)^2)

x(sq_in) = x(sq_in)/mag_position
y(sq_in) = y(sq_in)/mag_position
z(sq_in) = z(sq_in)/mag_position

u(sq_in) = -u(sq_in) + 2*(B1*(1+beta*x(sq_in)).*(x(sq_in)^2 - 1) + 0 + 0)
v(sq_in) = -v(sq_in) + 2*(B1*(1+beta*x(sq_in)).*(x(sq_in).*y(sq_in)) + 0 + 0)
w(sq_in) = -w(sq_in) + 2*(B1*(1+beta*x(sq_in)).*(x(sq_in).*z(sq_in)) + 0 + 0)


x(sq_in) = R1*x(sq_in)
y(sq_in) = R1*y(sq_in)
z(sq_in) = R1*z(sq_in)

figure(1)
quiver3(x,y,z,u,v,w,'filled', 'Marker', 'o', 'LineWidth', 1.8,'AutoScaleFactor',0.5)
daspect([1 1 1])
axis([xmin xmax ymin ymax zmin zmax])
hold on
pause()

x(sq_in) = x(sq_in) + u(sq_in)*dt2
y(sq_in) = y(sq_in) + v(sq_in)*dt2
z(sq_in) = z(sq_in) + w(sq_in)*dt2

figure(1)
quiver3(x,y,z,u,v,w,'filled', 'Marker', 'o', 'LineWidth', 1.8,'AutoScaleFactor',0.5)
daspect([1 1 1])
axis([xmin xmax ymin ymax zmin zmax])
hold on
pause()
