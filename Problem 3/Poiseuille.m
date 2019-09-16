clear all
close all
DIM=2;            %2D or 3D
Nitr=25000;   %Maximum number of iterations
ncx=51;           %Number of cells in x direction
ncy=51;            %Number of cells in y direction
nc_total=ncx*ncy; %Total number of cells in the grid
h=1;              %Cell length
h2=h/2;           %Half cell length
Lx=(ncx-1)*h;     %Length of the simulation box in x direction
Ly=(ncy-1)*h;     %Length of the simulation box in y direction
x_min=0;          %
y_min=0;
x_max=x_min+Lx;
y_max=y_min+Ly;
np_avg=10;   
N=(ncx-1)*(ncy-1)*np_avg;
%N=10000;      %Total number of particles in the simulation box
%np_avg=ceil(N/nc_total);   %Average number of particles per cell
mass=1;       %Mass of each particle
force=0.0002;      %Force on each particle in x direction
acc=force/mass;      %Acceleration in x direction        
mu=0;         %Mean of Gaussian distribution for initial velocity
sigma=1;      %Standard deviation of Gaussian distribution for initial velocity
lamda=0.1;    %Mean free path
flf
alpha=pi/2;    %Rotation angle for collision step
% cos_alpha=cos(alpha);
% sin_alpha=sin(alpha);
cos_alpha=0;
sin_alpha=1;
dt=lamda/sigma;      %Duration of each time step
% 
% 
%INITIALIZATION
x=x_min+rand(1,N)*Lx;%x coordinates of the particles
y=y_min+rand(1,N)*Ly;%y coordinates of the particles
u=normrnd(mu,sigma,[1 N]);%x components of velocites of the particles
v=normrnd(mu,sigma,[1 N]);%y components of velocites of the particles

%load('Init_Config_Poiseuille.mat')
u_rel=zeros(1,N);
v_rel=zeros(1,N);
u_rot=zeros(1,N);
v_rot=zeros(1,N);
location=zeros(1,N);%paticle cell locations
u_cm=zeros(1,nc_total);
v_cm=zeros(1,nc_total);
grid_shift_x=h*rand(1,Nitr)-h2;% why Nitr
grid_shift_y=h*rand(1,Nitr)-h2;
eta=zeros(1,N);
Velocity_data=zeros(ncy-1,Nitr);%?
Velocity_profile=zeros(1,ncy-1);%?
%momentum_x(1)=sum(u);
%momentum_y(1)=sum(v);
%energy(1)=sum(u.^2+v.^2); 
%figure(1);hold on;quiver(x,y,u,v,'-o')

for i=1:1:Nitr
    if(mod(i,50)==1)
        disp(i)
    end
    %Streaming step
    x=x+u*dt+0.5*acc*(dt^2);
    y=y+v*dt;
    u=u+acc*dt;
    %figure(i+1);hold on;quiver(x,y,u,v,'-o')
    %why not updated v - acc ony in x direction
    
    %Periodic boundary condition in y direction
%     y=y_min+mod(y-y_min,Ly);
    
    %Bounce back conditions in y direction
    top_escape=find(y>y_max);
    time_above_top=(y(top_escape)-y_max)./v(top_escape);
    y(top_escape)=2*y_max-y(top_escape);
    x(top_escape)=x(top_escape)-2*u(top_escape).*time_above_top+2*acc*(time_above_top).^2;
    u(top_escape)=-u(top_escape)+2*acc*time_above_top;
    v(top_escape)=-v(top_escape);
%     
    bottom_escape=find(y<y_min);
    time_below_bottom=(y(bottom_escape)-y_min)./v(bottom_escape);
    y(bottom_escape)=-y(bottom_escape);
    x(bottom_escape)=x(bottom_escape)-2*u(bottom_escape).*time_below_bottom+2*acc*(time_below_bottom).^2;
    u(bottom_escape)=-u(bottom_escape)+2*acc*time_below_bottom;
    v(bottom_escape)=-v(bottom_escape);
    
    
    %Periodoc boundary condition in x direction
    x=x_min+mod(x-x_min,Lx);
    
    %Allocate particles to cells
    ix=floor((x-x_min+h2-grid_shift_x(i))/h);
    iy=floor((y-y_min+h2-grid_shift_y(i))/h);
    location=iy*ncx+ix+1;
    
    %Calculation of velocity of centre of mass
    for j=1:1:nc_total
        neighbours=find(location==j);
        if(isempty(neighbours))
            continue
        end
        u_cm(j)=sum(u(neighbours))/length(neighbours);
        v_cm(j)=sum(v(neighbours))/length(neighbours);
    end
    u_rel=u-u_cm(location);
    v_rel=v-v_cm(location);
    %Collision step
    RandNumRot=2*randi([0,1],[1 nc_total])-1;
    u_rot=cos_alpha*u_rel-(sin_alpha*RandNumRot(location)).*v_rel;
    v_rot=(sin_alpha*RandNumRot(location)).*u_rel+cos_alpha*v_rel;
    u=u_cm(location)+u_rot;
    v=v_cm(location)+v_rot;
%     momentum_x(i+1)=sum(u);
%     momentum_y(i+1)=sum(v);
%     energy(i+1)=sum(u.^2+v.^2);

%     eta=floor((y-y_min)/h);
%     for l=1:1:ncy-1
%         same_eta=find(eta==l);
%         if(isempty(same_eta))
%             continue
%         end
%         Velocity_data(l,i)=sum(u(same_eta))/length(same_eta);
%     end
   
end
%  for k=1:1:ncy-1
%       Velocity_profile(k)=sum(Velocity_data(k,:))/Nitr;  
%  end