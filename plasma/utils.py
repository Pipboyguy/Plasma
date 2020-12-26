import torch
import torch.nn as nn 
import hiddenlayer as hl

def show_model(model:nn.Module):
    """
    Render graphviz representation of neural network model
    """
    hl_graph = hl.build_graph(model, torch.zeros([1, 3, 224, 224]))
    hl_graph.theme = hl.graph.THEMES["blue"].copy()
    
    return hl_graph

def plot_history(metrics_df):
    
    metrics_df_ = pd.melt(
        metrics_df,
        id_vars=["Epoch"],
        value_vars=list(set(metrics_df.columns) - set(["Epoch"])),
    )

    g = sns.lineplot(x="Epoch", y="value", hue="variable", data=metrics_df_)


    plt.show()

def get_last_n_layers(model: nn.Module, n_layers: int = 2) -> list:

    named_layers = sorted(
        list(
            set(
                [
                    re.findall("^[Ll]ayer\d", name)[0]
                    for name, _ in model.named_modules()
                    if re.search("^[Ll]ayer\d", name) is not None
                ]
            )
        )
    )

    #     get last n layers
    named_layers = named_layers[-n_layers:]

    return [getattr(model, name) for name in reversed(named_layers)], named_layers