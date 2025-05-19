package com.betool.ui;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.Border;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class BEToolUI extends JFrame {
    private JTabbedPane tabbedPane;
    private Font normalFont;
    private Font boldFont;

    public BEToolUI() {
        // 设置窗口标题和关闭操作
        setTitle("BE-Tool");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1100, 600);
        setLocationRelativeTo(null); // 窗口居中显示

        // 初始化字体
        normalFont = new Font("Arial", Font.PLAIN, 14);
        boldFont = new Font("Arial", Font.BOLD, 14);

        // 创建主面板
        try {
            initComponents();
        } catch (IllegalComponentStateException | IOException e) {
            e.printStackTrace();
        }
    }



    private void initComponents() throws IOException {
        // 创建选项卡面板
        tabbedPane = new JTabbedPane();

        // 添加ChangeListener监听选项卡切换
        tabbedPane.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                updateTabTitles();
            }
        });

        // 添加四个模块选项卡（全部添加图标）
        tabbedPane.addTab("Home", createIcon("home.png"), createIntroductionPanel());
        tabbedPane.addTab("sgRNA Design", createIcon("sgrna.png"), createSgRNAPanel());
        tabbedPane.addTab("Base Editor", createIcon("editor.png"), createBaseEditorPanel());
        tabbedPane.addTab("BE Lib", createIcon("library.png"), createBELibPanel());

        // 初始化选项卡标题样式
        updateTabTitles();

        // 将选项卡面板添加到窗口
        add(tabbedPane, BorderLayout.CENTER);
    }

    // 创建图标方法（修复版）
    private ImageIcon createIcon(String fileName) {
        try {
            // 从当前类路径加载图标
            java.net.URL imgURL = getClass().getResource(fileName);
            if (imgURL != null) {
                System.out.println("成功加载图标: " + fileName);
                // 调整图标大小（可选）
                ImageIcon icon = new ImageIcon(imgURL);
                return new ImageIcon(icon.getImage().getScaledInstance(
                        16, 16, Image.SCALE_SMOOTH));
            } else {
                System.err.println("找不到图标文件: " + fileName);
                return null;
            }
        } catch (Exception e) {
            System.err.println("加载图标时出错: " + e.getMessage());
            return null;
        }
    }

    // 更新选项卡标题（保留图标）
    private void updateTabTitles() {
        for (int i = 0; i < tabbedPane.getTabCount(); i++) {
            // 获取原有图标
            Icon icon = tabbedPane.getIconAt(i);

            // 创建包含图标和文本的标签
            JLabel label = new JLabel(tabbedPane.getTitleAt(i), icon, JLabel.LEFT);

            // 设置间距（可选）
            label.setBorder(BorderFactory.createEmptyBorder(0, 2, 0, 5));

            if (i == tabbedPane.getSelectedIndex()) {
                // 当前选中的选项卡：蓝色、加粗
                label.setForeground(Color.BLUE);
                label.setFont(boldFont);
            } else {
                // 未选中的选项卡：黑色、普通字体
                label.setForeground(Color.BLACK);
                label.setFont(normalFont);
            }

            // 设置自定义标签为选项卡组件
            tabbedPane.setTabComponentAt(i, label);
        }
    }

    private JPanel createIntroductionPanel() throws IOException {
        // 使用背景面板
        JPanel panel = new BackgroundPanel("BETool.jpg", true);

        // 添加介绍内容
        JTextArea introText = new JTextArea(
                "Base Editing Tool\n\n" +
                        "This tool provides comprehensive functions for base editing research, " +
                        "including sgRNA design, base editor selection, and library management.\n\n" +
                        "Key features:\n" +
                        "- sgRNA design with high specificity prediction\n" +
                        "- Comprehensive database of base editors\n" +
                        "- Library design and analysis tools\n" +
                        "- Off-target prediction and minimization",
                10, 30
        );
        introText.setEditable(false);
        introText.setLineWrap(true);
        introText.setWrapStyleWord(true);
        introText.setFont(new Font("Arial", Font.PLAIN, 14));
        introText.setOpaque(false); // 确保文本区域透明

        // 添加标题标签
        JLabel titleLabel = new JLabel("Welcome to Base Editing Tool");
        titleLabel.setFont(new Font("Arial", Font.BOLD, 20));
        titleLabel.setHorizontalAlignment(JLabel.CENTER);
        titleLabel.setForeground(Color.DARK_GRAY);
        titleLabel.setOpaque(false);

        // 创建透明面板放置文本内容
        JPanel textPanel = new JPanel(new BorderLayout());
        textPanel.setOpaque(false);
        textPanel.add(titleLabel, BorderLayout.NORTH);
        textPanel.add(new JScrollPane(introText), BorderLayout.CENTER);

        // 添加到背景面板
       // panel.add(textPanel, BorderLayout.CENTER);

        return panel;
    }

    private JPanel createSgRNAPanel() {
        // 使用背景面板（透明）
        JPanel panel = new BackgroundPanel("backgroud.png", true);
        panel.setLayout(new BorderLayout());


        SgRNADesignPanel SgRNADesignPanel = new SgRNADesignPanel();
        panel.add(SgRNADesignPanel);
        return panel;
    }

    private JPanel createBaseEditorPanel() {
        // 使用背景面板（透明）
        JPanel panel = new BackgroundPanel("backgroud.png", true);
        panel.setLayout(new BorderLayout());


        BEditorPanel beditorPanel = new BEditorPanel();
        panel.add(beditorPanel);
        return panel;
    }

    private JPanel createBELibPanel() {
        // 使用背景面板（透明）
        JPanel panel = new BackgroundPanel("backgroud.png", true);
        panel.setLayout(new BorderLayout());


        BELibPanel beLibPanel = new BELibPanel();
        panel.add(beLibPanel);
        return panel;
    }

    // 自定义背景面板类
    private static class BackgroundPanel extends JPanel {
        private BufferedImage backgroundImage;
        private final boolean scaleToFit;

        public BackgroundPanel(String imageName, boolean scaleToFit) {
            this.scaleToFit = scaleToFit;
            try {
                // 加载背景图
                backgroundImage = ImageIO.read(BEToolUI.class.getResource(imageName));
                if (backgroundImage == null) {
                    throw new IOException("无法加载背景图: " + imageName);
                }
            } catch (IOException e) {
                System.err.println("加载背景图时出错: " + e.getMessage());
                // 设置默认背景色
                setBackground(Color.LIGHT_GRAY);
            }
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (backgroundImage != null) {
                Graphics2D g2d = (Graphics2D) g.create();
                g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BILINEAR);

                if (scaleToFit) {
                    // 缩放图片以适应面板
                    g2d.drawImage(backgroundImage, 0, 0, getWidth(), getHeight(), this);
                } else {
                    // 原始大小绘制
                    g2d.drawImage(backgroundImage, 0, 0, this);
                }

                g2d.dispose();
            }
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            BEToolUI app = new BEToolUI();
            app.setVisible(true);
        });
    }
}