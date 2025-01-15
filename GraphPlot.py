import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
def Graphs(xval, yval):
    # Create the tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Linear", "Polynomial", "Logarithmic", "Exponential", "Histogram"])
    with tab1:
        Linear(xval,yval)
    with tab2:
        degree = st.slider('Select Polynomial Degree', min_value=1, max_value=10, value=2)
        Polynomial(xval, yval, degree)
    with tab3:
        Log(xval,yval)
    with tab4:
        Exp(xval,yval)
    with tab5:
        Hist(xval,yval)
#%%
def Linear(x,y):
        # Convert x and y to numpy arrays (in case they're lists)
        x = np.array(x)
        y = np.array(y)
    
        # Perform linear regression (find the best fit line)
        # np.polyfit returns the slope and the intercept of the best fit line
        m, b = np.polyfit(x, y, 1)  # 1 indicates a linear fit (degree 1)
    
        # Generate y values for the best fit line
        yfit = m * x + b
        
        errors = np.abs(y - yfit)
        avg_error = np.mean(errors)
        max_error = np.max(errors)

        # Create the plot (8x6 inches)
        plt.figure(figsize=(8, 6))
    
        # Plot original data points
        plt.scatter(x, y, color='blue', label='Data Points')
    
        # Plot the best fit line
        plt.plot(x, yfit, color='red', label=f'Best Fit Line: y = {m:.2f}x {"+ " if b >= 0 else "- "}{abs(b):.2f}') # 2 decimal points & proper formatting to change + - to -
    
        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Best Fit Linear Line')
        
        plt.legend(title=f'Avg Error = {avg_error:.2f}\nMax Error = {max_error:.2f}')

        # Show the plot
        plt.grid(True)
        st.pyplot(plt)
#%%
def Polynomial(x, y, degree):
    x = np.array(x)
    y = np.array(y)
    
    coefficients = np.polyfit(x, y, degree)  # Fit a polynomial of the specified degree
    
    # Create a smoother range of x values for plotting the polynomial curve
    xfit = np.linspace(min(x), max(x), 500)
    
    # Generate y values for the best fit polynomial
    yfit = np.polyval(coefficients, xfit)
    
    # Generate predicted y values for the original x values
    y_pred = np.polyval(coefficients, x)
    
    # Calculate errors (absolute differences between observed and predicted values)
    errors = np.abs(y - y_pred)
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Create the plot (8x6 Inches)
    plt.figure(figsize=(8, 6))
    
    # Plot original data points
    plt.scatter(x, y, color='blue', label='Data Points')
    
    # Plot the best fit polynomial
    poly_eqn = ' + '.join([f'{coef:.2f}x^{degree-i}' if i < degree else f'{coef:.2f}' 
                           for i, coef in enumerate(coefficients)])
    plt.plot(xfit, yfit, color='red', label=f'Best Fit Polynomial: {poly_eqn}')
    
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Best Fit Polynomial (Degree {degree})')
    plt.legend(title=f'Avg Error = {avg_error:.2f}\nMax Error = {max_error:.2f}') # 2 decimal place formating
    
    # Show the plot
    plt.grid(True)
    st.pyplot(plt)


#%%
def Log(x,y):
    # Ensure that x values are positive, as log(x) is undefined for non-positive values
    if any(val <= 0 for val in x):
        st.error("X values must be > 0 for a logarithmic fit.")
        return None
    x = np.array(x)
    y = np.array(y)
    logx = np.log(x)   
    # Perform linear regression (fit a line to log(x) and y) 
    # Able to use polyfit here thanks to the source above. 
    a, b = np.polyfit(logx, y, 1)
    # Generate x values for the fitted curve (using original x values)
    xfit = np.linspace(min(x), max(x), 500)
    # Apply the logarithmic transformation to the xfit values
    logxfit = np.log(xfit)
    # Generate y values for the best fit line
    yfit = a * logxfit + b
    
    logx_pred = np.log(x)
    y_pred = a * logx_pred + b
    errors = np.abs(y - y_pred)
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Create the plot (8x6 inches)
    plt.figure(figsize=(8, 6))
    # plot points and Curve
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.plot(xfit, yfit, color='red', label=f'Best Fit Logarithmic: y = {a:.2f}log(x) {"+" if b >= 0 else ""}{b:.2f}') # proper formatting to change + - to -
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Best Fit Logarithmic Curve')
    plt.legend(title=f'Avg Error = {avg_error:.2f}\nMax Error = {max_error:.2f}') # 2 decimal points
    plt.grid(True)
    st.pyplot(plt)
#%%
def Exp(x, y):
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Initialize C (in case no transformation is needed)
    C = 0
    
    # Check if there are any non-positive y values
    if np.any(y <= 0):
        # Find the index of the x value closest to zero
        index_at_zero = np.argmin(np.abs(x))  # Find the index closest to x = 0
        C = y[index_at_zero]  # The corresponding y value is the vertical shift (C)
        """
        Transform y by subtracting C and adding 1 (shifting the curve to avoid negative values)
        Increased by 1 because just like excel it is curve of best fit with ONLY
        VERTICAL SHIFTS AND NOT HORIZONTAL ONES 
        """
        # Transform y by subtracting C and adding 1 (shift the curve to avoid negative values)
        y_transformed = y - C + 1
        
        # Check if transformed y values are positive (valid log transformation)
        if np.any(y_transformed <= 0):
            st.error("Transformed Y values must be positive (greater than zero) for a valid log transformation.")
            return None
    else:
        # If y values are already positive, no transformation is necessary
        y_transformed = y
    
    # Take the natural log of the transformed y values
    logy = np.log(y_transformed)
    
    # Perform linear regression (fit a line to x and log(y_transformed))
    a, b = np.polyfit(x, logy, 1)
    
    # Generate x values for the fitted curve
    xfit = np.linspace(min(x), max(x), 500)
    
    # Apply the exponential function to the fitted line and recover y
    yfit = np.exp(a * xfit + b) + C - 1  # Re-add the vertical shift C after applying the exponential function
    
    y_pred = np.exp(a * x + b) + C - 1  # Predicted y values with the vertical shift (C - 1)
    
    # Calculate errors
    errors = np.abs(y - y_pred)  # Errors in original scale
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot the original data points and the fitted exponential curve
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.plot(xfit, yfit, color='red', label=f'Best Fit Exponential: y = {np.exp(b):.2f} * e^{a:.2f}x {"+" if (C - 1) >= 0 else ""}{C - 1:.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Best Fit Exponential Curve')
    
    # Add the equation in the legend or as text on the plot
    plt.legend(title=f'Avg Error = {avg_error:.2f}\nMax Error = {max_error:.2f}')
    
    # Display the grid and plot
    plt.grid(True)
    st.pyplot(plt)

#%%
def Hist(x, y):
    x = np.array(x)
    y = np.array(y)

    # Ask user to choose which list to plot
    user_choice = st.radio("Which list would you like to plot?", ('x', 'y'))

    # Select the appropriate data based on user input
    if user_choice == 'x':
        data = x
    else:
        data = y
    
    # Plot the histogram with a fixed number of bins to avoid extra frequency
    plt.figure(figsize=(8, 6))  # Set figure size
    
    n, bins, patches = plt.hist(data, bins=8, edgecolor='black', color='blue', align='mid', rwidth=1.0)
    plt.title(f"Histogram of List {user_choice.upper()}")
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    
    # Calculate mean and standard deviation manually
    mean = np.mean(data)
    std = np.std(data)

    # Generate x values for the bell curve (normal distribution)
    xmin, xmax = plt.xlim()  # Get the x-axis limits of the histogram
    x_vals = np.linspace(xmin, xmax, 100)  # Create a range of values for the bell curve

    # Calculate the normal distribution values using the formula
    p_vals = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mean) / std) ** 2)

    # Scale the bell curve to match the height of the histogram
    plt.plot(x_vals, p_vals * np.max(n) / np.max(p_vals), 'r-', label=f'Fit: Normal Distribution\nμ={mean:.2f}, σ={std:.2f}')
    
    # Show the legend
    plt.legend(loc='best')
    
    # Show the plot
    plt.grid(True)
    st.pyplot(plt)
#%%
# Title centered in HTML
st.markdown("""
    <h1 style='text-align: center; color: black;'>Desmos at Home</h1>
""", unsafe_allow_html=True)
input_method = st.radio("Choose an input method:", ("Upload CSV", "Manual Input"))

# Check if a file has been uploaded
if input_method == "Upload CSV":
    file = st.file_uploader('Enter a CSV File (Only 2 Columns in X, Y order)** (First Row is not Counted (for titles))')

    if file is not None:
        # If the file is uploaded, read it into a DataFrame
        df = pd.read_csv(file, header=None)
        st.write("File successfully uploaded!")

        # Checking if the file contains exactly 2 columns
        if df.shape[1] == 2:
            try:
                # Convert the columns to numeric values, coercing errors to NaN (if any)
                df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # Convert X column to numeric
                df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # Convert Y column to numeric

                # Check for NaN values which indicate non-numeric data
                if df.isna().any().any():
                    st.error("The CSV file contains corrupt data (non-numeric values, or mismatched column length).")
                else:
                    # Split columns into lists 
                    xval = df.iloc[:, 0].tolist()  # First column (X values)
                    yval = df.iloc[:, 1].tolist()  # Second column (Y values)

                    # Display X and Y values
                    st.text(f"X values: {xval}")
                    st.text(f"Y values: {yval}")
                    
                    # Calling the Graph function (replace with your actual graphing function)
                    Graphs(xval, yval)

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
        else:
            st.write("The CSV file does not satisfy Conditions. Try Another File")
# Manually input
else:
    # Initialize session state for X and Y values if they do not exist
    if 'xval' not in st.session_state:
        st.session_state.xval = []

    if 'yval' not in st.session_state:
        st.session_state.yval = []

    # Input field for adding new X values
    new_inputX = st.text_input(label='Enter X Values 1 by 1 or Separated by a comma', key='x_input')

    # Input field for adding new Y values
    new_inputY = st.text_input(label='Enter Y Values 1 by 1 or Separated by a comma', key='y_input')

    # Button to add X values to the list
    if st.button('Add X Values'):
        if new_inputX:  # Ensure there's input
            # Try converting each value to integer, allowing for negative numbers
            new_valuesX = []
            for value in new_inputX.split(','):
                try:
                    # Strip spaces and convert to integer
                    new_valuesX.append(float(value.strip()))
                except ValueError:
                    continue  # Ignore invalid entries that can't be converted
            st.session_state.xval.extend(new_valuesX)  # Modify session state directly
            st.success("X values added successfully!")

    # Button to add Y values to the list
    if st.button('Add Y Values'):
        if new_inputY:  # Ensure there's input
            # Try converting each value to integer, allowing for negative numbers
            new_valuesY = []
            for value in new_inputY.split(','):
                try:
                    # Strip spaces and convert to integer
                    new_valuesY.append(float(value.strip()))
                except ValueError:
                    continue  # Ignore invalid entries that can't be converted
            st.session_state.yval.extend(new_valuesY)  # Modify session state directly
            st.success("Y values added successfully!")

    # Assign the session state lists to local variables after processing input
    xval = st.session_state.xval
    yval = st.session_state.yval

    # Display the current X and Y values
    st.text("Current X Values: " + str(xval))
    st.text("Current Y Values: " + str(yval))
    
    if len(xval) != len(yval): # want same # of lists (col)
        st.error("Error: The number of X values and Y values do not match. Please ensure both lists have the same length.")
    elif len(xval) == len(yval) >=1:
        #Calling my Graph types and Graphing Fn
        Graphs(xval,yval)
#%%
"""
Works Optimally in Light Mode on the Web Browser

A special thanks to : 
https://www.cns.nyu.edu/~david/courses/perceptionLab/Handouts/Fitting%20Functions.pdf
Page 6 for finding the graph straightened functions needed to plot the graphs with only polyfit

Note:
Vertical transpositions work for all graphs, but horizontal stretches/compressions or inverses
will NOT work due to my restrictions
"""