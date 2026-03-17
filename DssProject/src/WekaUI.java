import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import javax.swing.*;
import javax.swing.border.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.JTableHeader;
import java.awt.*;
import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.swing.plaf.basic.BasicComboBoxUI;


public class WekaUI extends JFrame {

    private final Weka weka;

    // Modern color scheme
    private static final Color PRIMARY = new Color(99, 102, 241);      // Indigo
    private static final Color PRIMARY_DARK = new Color(79, 70, 229);
    private static final Color SECONDARY = new Color(236, 72, 153);    // Pink
    private static final Color SUCCESS = new Color(34, 197, 94);
    private static final Color BG_MAIN = new Color(17, 24, 39);        // Dark gray
    private static final Color BG_CARD = new Color(31, 41, 55);
    private static final Color BG_INPUT = new Color(55, 65, 81);
    private static final Color TEXT_PRIMARY = new Color(243, 244, 246);
    private static final Color TEXT_SECONDARY = new Color(156, 163, 175);
    private static final Color BORDER_COLOR = new Color(75, 85, 99);

    // UI Components
    private final JTextField txtPath = createStyledTextField();
    private final JButton btnBrowse = createModernButton("📁 Browse Dataset", PRIMARY);
    private final JButton btnRun = createModernButton("▶ Run Analysis", SECONDARY);

    private final JLabel lblDatasetInfo = createStyledLabel("No dataset loaded", 14, false);
    private final JLabel lblBest = createStyledLabel("⭐ Best Model: Not evaluated yet", 14, true);

    private final JProgressBar progress = createModernProgress();

    private final DefaultTableModel tableModel = new DefaultTableModel(
            new Object[]{"Algorithm", "Preprocessing", "✓", "Total", "Accuracy", "Status"}, 0
    ) {
        @Override public boolean isCellEditable(int row, int col) { return false; }
    };
    private final JTable table = new JTable(tableModel);

    private final JPanel inputPanel = new JPanel();
    private final JButton btnDiscover = createModernButton("🔮 Predict Class", SUCCESS);
    private final JLabel lblPrediction = createStyledLabel("Prediction: -", 16, true);

    // State
    private File selectedFile;
    private Instances data;
    private Instances header;
    private final Map<Integer, JComponent> inputComponents = new HashMap<>();

    public WekaUI(Weka weka) {
        super("WEKA ML Classifier Studio");
        this.weka = weka;

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1280, 800);
        setLocationRelativeTo(null);

        btnRun.setEnabled(false);
        btnDiscover.setEnabled(false);

        buildModernUI();
        wireActions();

        // Set dark theme
        getContentPane().setBackground(BG_MAIN);
    }

    private void buildModernUI() {
        JPanel root = new JPanel(new BorderLayout(20, 20));
        root.setBackground(BG_MAIN);
        root.setBorder(new EmptyBorder(20, 20, 20, 20));

        // Header Section
        JPanel header = createHeaderPanel();

        // Main Content
        JPanel content = new JPanel(new GridBagLayout());
        content.setBackground(BG_MAIN);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.BOTH;
        gbc.weightx = 1.0;
        gbc.insets = new Insets(0, 0, 15, 0);

        // Dataset info card
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.weighty = 0;
        content.add(createInfoCard(), gbc);

        // Progress bar
        gbc.gridy = 1;
        content.add(createProgressCard(), gbc);

        // Split pane with table and inputs
        gbc.gridy = 2;
        gbc.weighty = 1.0;
        content.add(createMainSplitPane(), gbc);

        // Prediction footer
        gbc.gridy = 3;
        gbc.weighty = 0;
        content.add(createPredictionCard(), gbc);

        root.add(header, BorderLayout.NORTH);
        root.add(content, BorderLayout.CENTER);

        setContentPane(root);
    }

    private JPanel createHeaderPanel() {
        JPanel panel = new JPanel(new BorderLayout(15, 0));
        panel.setBackground(BG_MAIN);
        panel.setBorder(new EmptyBorder(0, 0, 20, 0));

        JLabel title = new JLabel("🤖 WEKA ML Studio");
        title.setFont(new Font("Segoe UI", Font.BOLD, 28));
        title.setForeground(TEXT_PRIMARY);

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 10, 0));
        buttonPanel.setBackground(BG_MAIN);
        buttonPanel.add(btnBrowse);
        buttonPanel.add(btnRun);

        panel.add(title, BorderLayout.WEST);
        panel.add(buttonPanel, BorderLayout.EAST);

        return panel;
    }

    private JPanel createInfoCard() {
        JPanel card = createCard();
        card.setLayout(new BorderLayout(10, 10));

        JLabel icon = new JLabel("📊");
        icon.setFont(new Font("Segoe UI", Font.PLAIN, 24));

        JPanel textPanel = new JPanel(new GridLayout(2, 1, 0, 5));
        textPanel.setBackground(BG_CARD);

        JLabel pathLabel = createStyledLabel("Dataset Path:", 12, false);
        pathLabel.setForeground(TEXT_SECONDARY);

        textPanel.add(pathLabel);
        textPanel.add(txtPath);

        card.add(icon, BorderLayout.WEST);
        card.add(textPanel, BorderLayout.CENTER);
        card.add(lblDatasetInfo, BorderLayout.SOUTH);

        return card;
    }

    private JPanel createProgressCard() {
        JPanel card = createCard();
        card.setLayout(new BorderLayout(10, 5));

        JLabel label = createStyledLabel("Analysis Progress", 12, false);
        label.setForeground(TEXT_SECONDARY);

        card.add(label, BorderLayout.NORTH);
        card.add(progress, BorderLayout.CENTER);

        return card;
    }

    private JSplitPane createMainSplitPane() {
        // Results table
        JPanel tablePanel = createCard();
        tablePanel.setLayout(new BorderLayout());

        JLabel tableTitle = createStyledLabel("📈 Model Results", 14, true);
        tableTitle.setBorder(new EmptyBorder(0, 0, 10, 0));

        styleTable();
        JScrollPane tableScroll = new JScrollPane(table);
        tableScroll.setBorder(null);
        tableScroll.getViewport().setBackground(BG_CARD);

        tablePanel.add(tableTitle, BorderLayout.NORTH);
        tablePanel.add(tableScroll, BorderLayout.CENTER);
        tablePanel.add(lblBest, BorderLayout.SOUTH);

        // Input panel
        JPanel inputCard = createCard();
        inputCard.setLayout(new BorderLayout());

        JLabel inputTitle = createStyledLabel("🎯 Feature Input", 14, true);
        inputTitle.setBorder(new EmptyBorder(0, 0, 10, 0));

        inputPanel.setLayout(new GridBagLayout());
        inputPanel.setBackground(BG_CARD);

        JScrollPane inputScroll = new JScrollPane(inputPanel);
        inputScroll.setBorder(null);
        inputScroll.getViewport().setBackground(BG_CARD);

        inputCard.add(inputTitle, BorderLayout.NORTH);
        inputCard.add(inputScroll, BorderLayout.CENTER);

        JSplitPane split = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, tablePanel, inputCard);
        split.setResizeWeight(0.6);
        split.setDividerSize(8);
        split.setBorder(null);
        split.setBackground(BG_MAIN);

        return split;
    }

    private JPanel createPredictionCard() {
        JPanel card = createCard();
        card.setLayout(new BorderLayout(15, 0));

        card.add(btnDiscover, BorderLayout.WEST);

        JPanel resultPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 15, 0));
        resultPanel.setBackground(BG_CARD);
        resultPanel.add(lblPrediction);

        card.add(resultPanel, BorderLayout.CENTER);

        return card;
    }

    private JPanel createCard() {
        JPanel card = new JPanel();
        card.setBackground(BG_CARD);
        card.setBorder(BorderFactory.createCompoundBorder(
                new LineBorder(BORDER_COLOR, 1, true),
                new EmptyBorder(15, 15, 15, 15)
        ));
        return card;
    }

    private JButton createModernButton(String text, Color color) {
        JButton btn = new JButton(text);
        btn.setFont(new Font("Segoe UI", Font.BOLD, 13));
        btn.setBackground(color);
        btn.setForeground(Color.WHITE);
        btn.setFocusPainted(false);
        btn.setBorderPainted(false);
        btn.setBorder(new EmptyBorder(10, 20, 10, 20));
        btn.setCursor(new Cursor(Cursor.HAND_CURSOR));

        btn.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseEntered(java.awt.event.MouseEvent evt) {
                if (btn.isEnabled()) {
                    btn.setBackground(color.darker());
                }
            }
            public void mouseExited(java.awt.event.MouseEvent evt) {
                btn.setBackground(color);
            }
        });

        return btn;
    }

    private JTextField createStyledTextField() {
        JTextField tf = new JTextField();
        tf.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        tf.setBackground(BG_INPUT);
        tf.setForeground(TEXT_PRIMARY);
        tf.setCaretColor(TEXT_PRIMARY);
        tf.setBorder(new CompoundBorder(
                new LineBorder(BORDER_COLOR, 1, true),
                new EmptyBorder(8, 12, 8, 12)
        ));
        tf.setEditable(false);
        return tf;
    }

    private JLabel createStyledLabel(String text, int size, boolean bold) {
        JLabel lbl = new JLabel(text);
        lbl.setFont(new Font("Segoe UI", bold ? Font.BOLD : Font.PLAIN, size));
        lbl.setForeground(TEXT_PRIMARY);
        return lbl;
    }

    private JProgressBar createModernProgress() {
        JProgressBar pb = new JProgressBar(0, 100);
        pb.setStringPainted(true);
        pb.setFont(new Font("Segoe UI", Font.BOLD, 12));
        pb.setBackground(BG_INPUT);
        pb.setForeground(PRIMARY);
        pb.setBorder(new LineBorder(BORDER_COLOR, 1, true));
        pb.setPreferredSize(new Dimension(0, 35));
        return pb;
    }

    private void styleTable() {
        table.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        table.setRowHeight(32);
        table.setBackground(BG_CARD);
        table.setForeground(TEXT_PRIMARY);
        table.setGridColor(BORDER_COLOR);
        table.setSelectionBackground(PRIMARY.darker());
        table.setSelectionForeground(Color.WHITE);
        table.setShowGrid(true);
        table.setIntercellSpacing(new Dimension(1, 1));

        JTableHeader header = table.getTableHeader();
        header.setFont(new Font("Segoe UI", Font.BOLD, 12));
        header.setBackground(BG_INPUT);
        header.setForeground(TEXT_PRIMARY);
        header.setBorder(new LineBorder(BORDER_COLOR));

        DefaultTableCellRenderer centerRenderer = new DefaultTableCellRenderer();
        centerRenderer.setHorizontalAlignment(JLabel.CENTER);
        centerRenderer.setBackground(BG_CARD);
        centerRenderer.setForeground(TEXT_PRIMARY);

        for (int i = 2; i < 5; i++) {
            table.getColumnModel().getColumn(i).setCellRenderer(centerRenderer);
        }
    }

    private void wireActions() {
        btnBrowse.addActionListener(e -> onBrowse());
        btnRun.addActionListener(e -> onRun());
        btnDiscover.addActionListener(e -> onDiscover());
    }

    private void onBrowse() {
        JFileChooser chooser = new JFileChooser();
        chooser.setDialogTitle("Select WEKA Dataset");
        chooser.setFileFilter(new FileNameExtensionFilter("WEKA Datasets (*.arff, *.csv)", "arff", "csv"));

        int res = chooser.showOpenDialog(this);
        if (res != JFileChooser.APPROVE_OPTION) return;

        selectedFile = chooser.getSelectedFile();
        txtPath.setText(selectedFile.getAbsolutePath());

        resetUIState();
        setProgress(5, "Loading dataset...");

        try {
            data = weka.loadDataset(selectedFile);
            weka.validateDatasetForClassification(data);
            header = new Instances(data, 0);

            lblDatasetInfo.setText(String.format(
                    "✓ %d instances  •  %d features  •  Class: %s",
                    data.numInstances(), data.numAttributes(), data.classAttribute().name()
            ));

            buildDynamicInputPanel(header);
            btnRun.setEnabled(true);
            setProgress(0, "Ready to analyze");
        } catch (Exception ex) {
            showError("Failed to load dataset", ex);
            setProgress(0, "Error occurred");
        }
    }

    private void onRun() {
        if (data == null) return;

        btnRun.setEnabled(false);
        btnBrowse.setEnabled(false);
        btnDiscover.setEnabled(false);

        tableModel.setRowCount(0);
        lblBest.setText("⭐ Best Model: Analyzing...");
        lblPrediction.setText("Prediction: -");

        SwingWorker<Weka.RunResult, Void> worker = new SwingWorker<>() {
            @Override
            protected Weka.RunResult doInBackground() throws Exception {
                return weka.runAllModels(data, (p, msg) -> WekaUI.this.setProgress(p, msg));
            }

            @Override
            protected void done() {
                try {
                    Weka.RunResult r = get();
                    fillResults(r.rows);

                    lblBest.setText(String.format(
                            "⭐ Best: %s  •  Accuracy: %.2f%%  •  Correct: %d/%d",
                            r.bestRow.algo, r.bestRow.accuracy, r.bestRow.correct, r.bestRow.total
                    ));

                    btnDiscover.setEnabled(true);
                } catch (Exception ex) {
                    lblBest.setText("⭐ Best Model: Error occurred");
                    showError("Evaluation failed", ex);
                } finally {
                    btnRun.setEnabled(true);
                    btnBrowse.setEnabled(true);
                    WekaUI.this.setProgress(100, "✓ Analysis complete!");
                }
            }
        };

        worker.execute();
    }

    private void onDiscover() {
        if (header == null || !weka.hasBestModel()) {
            JOptionPane.showMessageDialog(this, "Please run the analysis first.",
                    "No Model", JOptionPane.WARNING_MESSAGE);
            return;
        }

        try {
            Map<Integer, Object> values = readInputValues();
            Instance inst = weka.buildUserInstance(header, values);
            String label = weka.predictLabel(header, inst);

            // Build detailed prediction info
            StringBuilder details = new StringBuilder();
            details.append("<html><div style='font-family:Segoe UI; color:#F3F4F6;'>");
            details.append("<b style='font-size:14px; color:#22C55E;'>🎯 Prediction: ").append(label).append("</b><br><br>");

            details.append("<b>Input Values:</b><br>");
            for (Map.Entry<Integer, Object> e : values.entrySet()) {
                Attribute attr = header.attribute(e.getKey());
                details.append("&nbsp;&nbsp;• <b>").append(attr.name()).append(":</b> ")
                        .append(e.getValue()).append("<br>");
            }

            details.append("</div></html>");

            lblPrediction.setText(details.toString());
            lblPrediction.setForeground(SUCCESS);
        } catch (Exception ex) {
            showError("Prediction failed", ex);
        }
    }

    private Map<Integer, Object> readInputValues() {
        Map<Integer, Object> values = new HashMap<>();

        for (Map.Entry<Integer, JComponent> e : inputComponents.entrySet()) {
            int idx = e.getKey();
            Attribute attr = header.attribute(idx);
            JComponent comp = e.getValue();

            if (attr.isNumeric()) {
                JTextField tf = (JTextField) comp;
                String txt = tf.getText().trim();
                if (txt.isEmpty()) {
                    throw new IllegalArgumentException("Please fill: " + attr.name());
                }
                values.put(idx, txt);
            } else {
                @SuppressWarnings("unchecked")
                JComboBox<String> cb = (JComboBox<String>) comp;
                Object sel = cb.getSelectedItem();
                if (sel == null) {
                    throw new IllegalArgumentException("Please select: " + attr.name());
                }
                values.put(idx, sel.toString());
            }
        }
        return values;
    }

    private void buildDynamicInputPanel(Instances head) {
        inputPanel.removeAll();
        inputComponents.clear();

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(8, 8, 8, 8);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1.0;

        int row = 0;

        for (int i = 0; i < head.numAttributes(); i++) {
            if (i == head.classIndex()) continue;

            Attribute attr = head.attribute(i);

            gbc.gridx = 0;
            gbc.gridy = row;
            gbc.weightx = 0.3;

            JLabel label = new JLabel(attr.name());
            label.setFont(new Font("Segoe UI", Font.BOLD, 12));
            label.setForeground(TEXT_PRIMARY);
            inputPanel.add(label, gbc);

            gbc.gridx = 1;
            gbc.weightx = 0.7;

            if (attr.isNumeric()) {
                JTextField tf = createStyledTextField();
                tf.setEditable(true);
                inputPanel.add(tf, gbc);
                inputComponents.put(i, tf);
            } else if (attr.isNominal()) {
                JComboBox<String> cb = new JComboBox<>();
                cb.setFont(new Font("Segoe UI", Font.PLAIN, 12));

                for (int v = 0; v < attr.numValues(); v++) cb.addItem(attr.value(v));
                styleComboBox(cb);
                inputPanel.add(cb, gbc);
                inputComponents.put(i, cb);
                // Force consistent colors for the "selected item" area
                cb.setEditable(true);
                JTextField editor = (JTextField) cb.getEditor().getEditorComponent();
                editor.setEditable(false);
                editor.setBackground(BG_INPUT);
                editor.setForeground(TEXT_PRIMARY);
                editor.setCaretColor(TEXT_PRIMARY);
                editor.setBorder(new EmptyBorder(8, 12, 8, 12));

                cb.setBackground(BG_INPUT);
                cb.setForeground(TEXT_PRIMARY);
                cb.setBorder(new CompoundBorder(
                        new LineBorder(BORDER_COLOR, 1, true),
                        new EmptyBorder(2, 2, 2, 2)
                ));
                cb.setOpaque(true);

                // Dropdown list colors
                cb.setRenderer(new DefaultListCellRenderer() {
                    @Override
                    public Component getListCellRendererComponent(JList<?> list, Object value, int index,
                                                                  boolean isSelected, boolean cellHasFocus) {
                        JLabel l = (JLabel) super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
                        l.setOpaque(true);
                        l.setBackground(isSelected ? PRIMARY_DARK : BG_INPUT);
                        l.setForeground(TEXT_PRIMARY);
                        l.setBorder(new EmptyBorder(6, 10, 6, 10));
                        return l;
                    }
                });

                inputPanel.add(cb, gbc);
                inputComponents.put(i, cb);
            }


            row++;
        }

        gbc.gridx = 0;
        gbc.gridy = row;
        gbc.gridwidth = 2;
        gbc.weighty = 1.0;
        gbc.fill = GridBagConstraints.BOTH;
        inputPanel.add(Box.createVerticalGlue(), gbc);

        inputPanel.revalidate();
        inputPanel.repaint();
    }

    private void fillResults(List<Weka.EvalRow> rows) {
        tableModel.setRowCount(0);
        for (Weka.EvalRow r : rows) {
            tableModel.addRow(new Object[]{
                    r.algo, r.prep, r.correct, r.total,
                    String.format("%.2f%%", r.accuracy), r.status
            });
        }
    }

    private void resetUIState() {
        tableModel.setRowCount(0);
        lblBest.setText("⭐ Best Model: Not evaluated yet");
        lblPrediction.setText("Prediction: -");
        btnRun.setEnabled(false);
        btnDiscover.setEnabled(false);
        setProgress(0, "Ready");
    }

    private void setProgress(int value, String msg) {
        SwingUtilities.invokeLater(() -> {
            progress.setValue(Math.max(0, Math.min(100, value)));
            progress.setString(msg);
        });
    }

    private void showError(String title, Exception ex) {
        ex.printStackTrace();
        JOptionPane.showMessageDialog(this, title + ":\n" + ex.getMessage(),
                title, JOptionPane.ERROR_MESSAGE);
    }
    private void styleComboBox(JComboBox<String> cb) {
        cb.setOpaque(true);
        cb.setBackground(BG_INPUT);
        cb.setForeground(TEXT_PRIMARY);

        cb.setBorder(new CompoundBorder(
                new LineBorder(BORDER_COLOR, 1, true),
                new EmptyBorder(6, 10, 6, 10)
        ));

        // Dropdown + selected item renderer
        cb.setRenderer(new DefaultListCellRenderer() {
            @Override
            public Component getListCellRendererComponent(JList<?> list, Object value, int index,
                                                          boolean isSelected, boolean cellHasFocus) {
                JLabel l = (JLabel) super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
                l.setOpaque(true);

                // index == -1 => combo'nun üstte görünen seçili kısmı
                if (index == -1) {
                    l.setBackground(BG_INPUT);
                    l.setForeground(TEXT_PRIMARY);
                } else {
                    l.setBackground(isSelected ? PRIMARY_DARK : BG_INPUT);
                    l.setForeground(TEXT_PRIMARY);
                }
                l.setBorder(new EmptyBorder(6, 10, 6, 10));
                return l;
            }
        });

        // Force paint for the selected-value background (Windows LAF fix)
        cb.setUI(new BasicComboBoxUI() {
            @Override
            public void paintCurrentValueBackground(Graphics g, Rectangle bounds, boolean hasFocus) {
                g.setColor(BG_INPUT);
                g.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
            }
        });
    }

}