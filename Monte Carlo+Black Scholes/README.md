# Option Pricing Calculator

This project implements an Option Pricing Calculator using the Black-Scholes model and Monte Carlo simulation. It provides a web-based interface built with Streamlit for easy interaction and visualization of option pricing and related metrics.

## Features

- Black-Scholes model implementation for option pricing and Greeks calculation
- Monte Carlo simulation for option pricing
- Interactive web interface for inputting parameters and viewing results
- Visualization of option prices, Greeks, and price paths

## Project Structure

- `blackscholes.py`: Contains the `BlackScholes` class for option pricing and Greeks calculation
- `monte_carlo.py`: Implements Monte Carlo simulation for option pricing
- `streamlit_app.py`: The main Streamlit application for the user interface
- `requirements.txt`: Lists all the Python dependencies required for the project

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/option-pricing-calculator.git
   cd option-pricing-calculator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the Streamlit app:


This will start the local server and open the application in your default web browser.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Black-Scholes model implementation is based on standard financial formulas
- The Monte Carlo simulation is a simplified version for educational purposes
- Streamlit for providing an excellent framework for building data applications
