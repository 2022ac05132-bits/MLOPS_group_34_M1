from model import load_data, train_model, save_model

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Save the model
    save_model(model, "models/model.pkl")
    print("Model trained and saved to models/model.pkl")

if __name__ == "__main__":
    main()
