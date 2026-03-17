import javax.swing.*;

public class Main {
    public static void main(String[] args) {
        try { UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName()); }
        catch (Exception ignored) {}

        SwingUtilities.invokeLater(() -> {
            Weka weka = new Weka();
            WekaUI ui = new WekaUI(weka);
            ui.setVisible(true);
        });
    }
}
