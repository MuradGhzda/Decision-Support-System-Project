# Decision Support System

A Java-based Decision Support System (DSS) that helps users evaluate options and make informed decisions based on defined criteria and scoring logic.

---

## What It Does

A Decision Support System takes a set of alternatives (options) and a set of criteria, then scores and ranks them to help a user arrive at the best choice. This project implements that process in Java, guiding the user through input, evaluation, and a final recommendation.

**Example use cases:**
- Choosing between job offers based on salary, location, and benefits
- Ranking products based on price, quality, and features
- Selecting a supplier based on cost, reliability, and delivery time

---

## How It Works

1. The user defines the **alternatives** — the options being compared
2. The user defines the **criteria** — what matters in the decision
3. Each alternative is scored against each criterion
4. The system calculates a total weighted score for each alternative
5. The alternative with the highest score is recommended

---

## Requirements

- **Java** (JDK 8 or higher)
- An IDE such as [IntelliJ IDEA](https://www.jetbrains.com/idea/) or [Eclipse](https://www.eclipse.org/) (recommended)

---

## Getting Started

1. **Clone the repository**
   ```
   git clone https://github.com/MuradGhzda/Decision-Support-System-Project.git
   ```

2. **Open the project**  
   Open the `DssProject` folder in your IDE as a Java project.

3. **Build the project**  
   Let the IDE resolve dependencies and compile the source files.

4. **Run the application**  
   Run the main class to launch the application.

---

## Project Structure

```
├── DssProject/          # Main Java project folder
│   └── src/             # Java source files
├── .gitattributes
└── README.md
```

---

## Language & Platform

- **Language:** Java
- **Platform:** Desktop / Console
