clear all
close all
% DIM=2;            %2D or 3D
% Nitr=35000;   %Maximum number of iterations
% ncx=51;           %Number of cells in x direction
% ncy=51;            %Number of cells in y direction
% nc_total=ncx*ncy; %Total number of cells in the grid
% h=1;              %Cell length
% h2=h/2;           %Half cell length
% Lx=(ncx-1)*h;     %Length of the simulation box in x direction
% Ly=(ncy-1)*h;     %Length of the simulation box in y direction
% x_min=0;          %
% y_min=0;
% x_max=x_min+Lx;
% y_max=y_min+Ly;
% np_avg=10;   
% N=(ncx-1)*(ncy-1)*np_avg;
% 
% 
% mass=1;       %Mass of each particle
% force=0;      %Force on each particle in x direction
% acc=force/mass;      %Acceleration in x direction
% mu=0;         %Mean of Gaussian distribution for initial velocity
% sigma=1;      %Standard deviation of Gaussian distribution for initial velocity
% lamda=0.1;    %Mean free path
% alpha=pi/2;    %Rotation angle for collision step
% % cos_alpha=cos(alpha);
% % sin_alpha=sin(alpha);
% cos_alpha=0;
% sin_alpha=1;
% dt=lamda/sigma;      %Duration of each time step
% U=1;
% %INITIALIZATION
% x=x_min+rand(1,N)*Lx;%x coordinates of the particles
% y=y_min+rand(1,N)*Ly;%y coordinates of the particles
% u=normrnd(mu,sigma,[1 N]);%x components of velocites of the particles
% v=normrnd(mu,sigma,[1 N]);%y components of velocites of the particles

load('Couette_linked1.mat')
Nitr=35000;
u_rel=zeros(1,N);
v_rel=zeros(1,N);
u_rot=zeros(1,N);
v_rot=zeros(1,N);
location=zeros(1,N);%paticle cell locations

u_cm=zeros(1,nc_total);
v_cm=zeros(1,nc_total);

grid_shift_x=h*rand(1,Nitr)-h2;
grid_shift_y=h*rand(1,Nitr)-h2;
eta=zeros(1,N);
Velocity_data=zeros(ncy-1,Nitr);
Velocity_profile=zeros(1,ncy-1);

for i=1:1:Nitr
    if(mod(i,50)==1)
        disp(i)
    end
    
    %Streaming step
    x=x+u*dt;
    y=y+v*dt;
    
    %Periodic boundary condition in y direction
%     y=y_min+mod(y-y_min,Ly);
    
    %Bounce back conditions in y direction
    top_escape=find(y>y_max);
    time_above_top=(y(top_escape)-y_max)./v(top_escape);
    y(top_escape)=2*y_max-y(top_escape);
    x(top_escape)=x(top_escape)-2*(u(top_escape)-U).*time_above_top;
    u(top_escape)=-u(top_escape)+2*U;
    v(top_escape)=-v(top_escape);
     
    bottom_escape=find(y<y_min);
    time_below_bottom=(y(bottom_escape)-y_min)./v(bottom_escape);
    y(bottom_escape)=-y(bottom_escape);
    x(bottom_escape)=x(bottom_escape)-2*u(bottom_escape).*time_below_bottom;
    u(bottom_escape)=-u(bottom_escape);
    v(bottom_escape)=-v(bottom_escape);
    
    
    %Periodoc boundary condition in x direction
    x=x_min+mod(x-x_min,Lx);
    
    %Allocate particles to cells
    ix=floor((x-x_min+h2-grid_shift_x(i))/h);
    iy=floor((y-y_min+h2-grid_shift_y(i))/h);
    location=iy*ncx+ix+1;
    
    head=zeros(1,nc_total)-1;
    list=zeros(1,N);
    np=zeros(1,nc_total);
 for p=1:1:N
        np(location(p))=np(location(p))+1;
        list(p)=head(location(p));
        head(location(p))=p;
 end
     u_sum=zeros(1,nc_total);
     v_sum=zeros(1,nc_total);
     
    for j=1:1:nc_total
        
        k=head(j);
        while k~=-1
            u_sum(j)=u_sum(j)+u(k);
            v_sum(j)=v_sum(j)+v(k);
            k=list(k);
        end
        if np(j)~=0
            u_cm(j)=u_sum(j)/np(j);
            v_cm(j)=v_sum(j)/np(j);
        end
        
    end
    
    
    u_rel=u-u_cm(location);
    v_rel=v-v_cm(location);
    %Collision step
    RandNumRot=2*randi([0,1],[1 nc_total])-1;
    u_rot=cos_alpha*u_rel-(sin_alpha*RandNumRot(location)).*v_rel;
    v_rot=(sin_alpha*RandNumRot(location)).*u_rel+cos_alpha*v_rel;
    u=u_cm(location)+u_rot;
    v=v_cm(location)+v_rot;

    eta=floor((y-y_min)/h)+1;
    for m=1:1:ncy-1
        same_eta=find(eta==m);
        if(isempty(same_eta))
            continue
        end
        Velocity_data(m,i)=sum(u(same_eta))/length(same_eta);
    end
    
end
  for k=1:1:ncy-1
      Velocity_profile(k)=sum(Velocity_data(k,:))/Nitr;  
 end