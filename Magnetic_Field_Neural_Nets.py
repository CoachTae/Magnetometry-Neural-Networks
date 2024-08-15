import torch
import torch.nn as nn
import Support
import numpy as np

# TO-DO LIST:
    # Check if residuals instead of mse would be better for loss function
    # Visualization of predictions
        # Plot magnetic field?
    # Logging and Monitoring
        # Use TensorBoard to track model performance, loss values, etc
        # Aids in model development and tuning
        # Might be too much? See into its viability
    


class BaseNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super(BaseNN, self).__init__()
        
        # Set to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Define the initial layer (from input to first hidden layer)
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        
        # Define hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Define the output layer (from last hidden layer to output)
        self.out = nn.Linear(hidden_size, output_size)
        
        # Define the activation function
        self.activation = nn.Tanh()
    
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.out(x)
        return x


class MagneticFieldNN(BaseNN):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super(MagneticFieldNN, self).__init__(input_size, hidden_size, num_hidden_layers, output_size)

        # Define MSE loss object
        self.mse_loss_object = nn.MSELoss()
        
    def forward(self, x):
        x = super(MagneticFieldNN, self).forward(x)
        return x

    def compute_mse_loss(self, predictions, targets):
        # Compute Losses
        mse_loss = self.mse_loss_object(predictions, targets)

        return mse_loss

    def compute_div_loss(self, inputs):
        # Ensure inputs require gradients
        inputs.requires_grad_(True)

        # Get model predictions
        outputs = self.forward(inputs)

        # Calculate gradients (partial derivatives) for each component
        # Creates an array of shape (outputs.shape) of all 1 entries
            # Don't fully understand this line
            # Seems to mean that it cares about each gradient equally
                # So that none get ignored
        grad_outputs = torch.ones_like(outputs)
        gradients = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=grad_outputs, create_graph=True, only_inputs=True)[0]
        
        # Calculate divergence as the sum of partials
        divergence = gradients[:, 0] + gradients[:, 1] + gradients[:, 2]

        return torch.mean(divergence ** 2)

    def compute_curl_loss(self, inputs):
        # Ensure inputs require gradients
        inputs.requires_grad_(True)

        # Get model predictions
        outputs = self.forward(inputs)

         # Compute gradients for each output component
        grad_outputs_x = torch.autograd.grad(outputs[:, 0].sum(), inputs, create_graph=True, only_inputs=True, allow_unused=True)[0]
        grad_outputs_y = torch.autograd.grad(outputs[:, 1].sum(), inputs, create_graph=True, only_inputs=True, allow_unused=True)[0]
        grad_outputs_z = torch.autograd.grad(outputs[:, 2].sum(), inputs, create_graph=True, only_inputs=True, allow_unused=True)[0]

        # Construct each component of the curl
        curl_x = grad_outputs_z[:, 1] - grad_outputs_y[:, 2]
        curl_y = grad_outputs_x[:, 2] - grad_outputs_z[:, 0]
        curl_z = grad_outputs_y[:, 0] - grad_outputs_x[:, 1]

        # Compute the squared magnitude of the curl vector
        curl_magnitude_squared = curl_x**2 + curl_y**2 + curl_z**2

        # Return the average squared magnitude as the loss
        return torch.mean(curl_magnitude_squared)

    def compute_total_loss(self,
                           predictions, # Array of predicted values
                           targets, # Array of target values 
                           inputs, # Array of input values
                           lambdas): # list of 3 lambda values
        # Grab all 3 components of the total loss function
        mse_loss = self.compute_mse_loss(predictions, targets)
        div_loss = self.compute_div_loss(inputs)
        curl_loss = self.compute_curl_loss(inputs)

        total_loss = lambdas[0]*mse_loss + lambdas[1]*div_loss + lambdas[2]*curl_loss

        return total_loss

    def interior_scan(self,
                      num_points_per_scan, # Number of points per scan
                      num_scans, # Number of consecutive scans that must meet the threshold to end the scanning
                      region=1, # What region we're scanning (to avoid F coil)
                      threshold=1e-8, # Loss threshold that must be met to count as successful
                      optimizer=None, # None for initial scan. Optimizer from training method can pass in an argument here so that we don't reinitialize every time
                      int_scan_lr=1e-3, # Default learning rate for interior scan optimizer
                      device=None): # Make sure we're on the same device as model

        if device is None:
            device = self.device
        
        # Make sure model is on the correct device
        self.to(device)

        self.train() # Set model to training mode

        if optimizer == None:
            optimizer = torch.optim.Adam(self.parameters(), lr=int_scan_lr)

        scan_number = 0
        while scan_number < num_scans:
            points = Support.random_points(num_points_per_scan, region=region, device=device)
            
            points.requires_grad_(True) # Require gradients for input points
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward(points)

            # Compute losses
            div_loss = self.compute_div_loss(points)
            curl_loss = self.compute_curl_loss(points)
            total_loss = div_loss + curl_loss

            # Backward pass
            total_loss.backward()

            # Update parameters
            optimizer.step()

            # Find max loss
            max_loss = torch.max(total_loss)

            if max_loss < threshold:
                scan_number += 1
            else:
                scan_number = 0

    # "data" variable in the method call will differentiate different types of data
        # "Boundary" will do just the boundary of a cylinder within the specified region
        # Region 1 is the UDET region starting at z=3cm
    def train_model(self,
                    num_epochs,
                    num_points, # Number of points to train per epoch
                    train_split, # What fraction b/w 0 and 1 is used for training
                    validation_threshold, # What threshold suffices during validation
                    learning_rate=1e-3, # Learning rate for the optimizer
                    do_int_scan=True,
                    data='Boundary',
                    region=1): # Only relevant for boundary training
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scan_optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train() # Set the model to training mode

        if data == 'Boundary':
            top_cap, bot_cap, shell = Support.load_boundary(region=region)
        elif data == 'Axis':
            axis_data = Support.load_axis()
        
        for epoch in range(num_epochs):
            if data == 'Boundary':
                # Pull random points from the 3 cylinder surfaces
                sampled_points_top_cap, sampled_points_bot_cap, sampled_points_shell = Support.random_boundary_points(num_points, top_cap, bot_cap, shell)
                sampled_points = np.vstack((sampled_points_top_cap, sampled_points_bot_cap, sampled_points_shell))

            elif data == 'Axis':
                sampled_points = Support.random_axis_points(num_points, axis_data)

            # Split the data into training and validation sets
            split_index = int(train_split * num_points)
            training_points = sampled_points[:split_index, :]
            validation_points = sampled_points[split_index:, :]

            # Do interior scan
            if do_int_scan:
                self.interior_scan(num_points, num_scans=10, threshold=1e-8, optimizer=scan_optimizer)

            # Training phase
            training_loss = self.run_training(training_points, optimizer)

            # Validation phase
            validation_loss = self.run_validation(validation_points)

            if validation_loss < validation_threshold:
                print("Validation criteria met. Training has ended.")
                break

            percent_done = (epoch + 1)*100 / num_epochs

            print("We are ", percent_done, "% done with training.")
            print("Loss: ", training_loss)
            print("Validation Loss: ", validation_loss)
            print('\n')

    # This function is only used within the "train_model" method. It's listed separately for readability
    def run_training(self, training_points, optimizer, batch_size=32):
        # Shuffle data
        np.random.shuffle(training_points)

        # Convert training points to torch tensor
        training_points_tensor = torch.tensor(training_points, dtype=torch.float, device=self.device)

        # Split training data into inputs and targets
        inputs = training_points_tensor[:, :3] # Pulls x, y, and z
        targets = training_points_tensor[:, 3:] # Pulls Bx, By, and Bz

        # Initialize loss for this whole set of points
        running_loss = 0.0

        # Iterate over batches
        for i in range(0, len(training_points), batch_size):
            # Extract batch
            inputs_batch = inputs[i:i + batch_size]
            targets_batch = targets[i:i + batch_size]

            # Zero the optimizer's parameter gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = self.forward(inputs_batch)

            # Compute Loss
            lambdas = [1, 1, 1]
            total_loss = self.compute_total_loss(predictions, targets_batch, inputs_batch, lambdas)
            
            # Backwards pass and optimize
            total_loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += total_loss.item() * inputs_batch.size(0)

        # Compute avg loss for this epoch
        epoch_loss = running_loss / len(training_points)

        return epoch_loss

    def run_validation(self, validation_points, batch_size=32):
        self.eval() # Set to evaluation mode

        # Accumulate loss over the validation set
        validation_loss = 0.0
        total_points = 0

        validation_points_tensor = torch.tensor(validation_points, dtype=torch.float, device=self.device)

        with torch.no_grad(): # Disable gradient computation
            for i in range(0, len(validation_points_tensor), batch_size):
                # Extract batch from validation points
                inputs_batch = validation_points_tensor[i:i + batch_size, :3]
                targets_batch = validation_points_tensor[i:i + batch_size, 3:]

                # Forward pass
                predictions = self.forward(inputs_batch)

                # Compute loss
                mse_loss = self.compute_mse_loss(predictions, targets_batch)

                # Accumulate validation loss
                validation_loss += mse_loss.item() * len(inputs_batch)
                total_points += len(inputs_batch)

        # Calculate avg loss over all points
        avg_validation_loss = validation_loss / total_points

        # Set model back to training mode
        self.train()

        return avg_validation_loss
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print("Model saved.")

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval() # Set model to evaluation mode after loading state
        print("Model loaded.")

    def save_optimizer(self, optimizer, path):
        torch.save(optimizer.state_dict(), path)
        print("Optimizer state saved.")

    def load_optimizer(self, optimizer, path):
        optimizer_state_dict = torch.load(path)
        optimizer.load_state_dict(optimizer_state_dict)

    def evaluate(self, inputs):
        # Ensure model is in evaluation mode
        self.eval()

        # Convert inputs to tensors
        inputs_tensor = torch.tensor(inputs, dtype=torch.float, device=self.device)

        # Disable gradient computation for evaluation for efficiency
        with torch.no_grad():
            predictions = self.forward(inputs_tensor)

        # Convert back to numpy array
        predictions_np = predictions.cpu().numpy()

        return predictions_np
