import pennylane as qml, torch
dev = qml.device("lightning.gpu", wires=12)
@qml.qnode(dev, interface="torch")
def circuit(theta):
    qml.templates.StronglyEntanglingLayers(theta, wires=range(12))
    return qml.expval(qml.PauliZ(0))
theta = torch.randn((1, 12, 3), requires_grad=True, device="cuda")
loss = circuit(theta)
loss.backward()
print("lightning.gpu OK — grad:", theta.grad.norm())
