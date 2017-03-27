def plot_pmf(
            t                  ,
            time_to_event      ,
            true_time_to_event ,
            is_censored        ,
            a                  ,
            b                  ,
            max_horizon        ,
            title = 'predicted Weibull pmf $p(t,s)$',
            lw = 1.0
        ):
    
    n = len(t)

    #[ax.axvline(x=k+1,lw=0.1,c='gray') for k in xrange(n-1)]
    
    resolution = max_horizon # Resolution on pdf graph, vertical height on pmf-graph

    # Discrete 
    pred = weibull_pmf(
                np.tile(np.linspace(0,max_horizon-1,resolution),(n,1)),
                np.tile(a.reshape(n,1),(1,resolution)),
                np.tile(b.reshape(n,1),(1,resolution))
                      )        

    ax.imshow(pred.T,origin='lower',interpolation='none',aspect='auto')
    ax.set_yticks([x*(resolution+0.0)/max_horizon for x in [0,max_horizon/2,max_horizon-1]])
    ax.set_yticklabels([0,max_horizon/2,max_horizon-1])
    ax.set_ylim(-0.5, resolution-0.5)
    ax.set_ylabel('steps ahead $s$')
    ax.set_title(title)

    def add_scaled_line(t,y,linestyle='solid',color='black'):
        # Shifts and scales y to fit on an imshow as we expect it to be, i.e passing through middle of a pixel
        scaled_y =((resolution+0.0)/max_horizon)*y
        ax.plot(t-0.5,scaled_y,lw=lw,linestyle=linestyle,drawstyle='steps-post',color=color,label = 'time to event') 
        # Adds last segment of steps-post that gets missing
        ax.plot([t[-1]-0.5,t[-1]+0.5],[scaled_y[-1],scaled_y[-1]],lw=lw,linestyle=linestyle,drawstyle='steps-post',color=color) 
        ax.set_xlim(-0.5, n-0.5)
    
    if true_time_to_event is not None:
        add_scaled_line(t,y=true_time_to_event,linestyle='solid')
    if time_to_event is not None:
        # todo fix bug where last uncensored ttestep is dotted if true_time_to_event is not None
        add_scaled_line(t[is_censored==0],y=time_to_event[is_censored==0],linestyle='solid')
        add_scaled_line(t[is_censored==1],y=time_to_event[is_censored==1],linestyle='dotted')

    ax.locator_params(axis='y',nbins=4) 
    ax.locator_params(axis='x',nbins=10)
#     for k in [0,1,2]:        
#         ax[k].set_xticks(ax[5].get_xticks()-0.5)    
#         ax[k].set_xticklabels(ax[5].get_xticks().astype(int))

#    ax[-1].set_xlabel('time')

    fig.tight_layout()

    return fig, ax