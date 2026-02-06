"""
Attention Visualization Module
Integrates with IzzyViz to create comparison heatmaps
"""
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Configure matplotlib for better font support (both Chinese and English)
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Add IzzyViz to path
izzyviz_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(izzyviz_path))

from izzyviz import (
    compare_two_attentions_with_circles,
    visualize_attention_self_attention
)


class AttentionVisualizer:
    """Handles attention visualization using IzzyViz"""

    def __init__(self, output_dir: str = '../outputs/attention_heatmaps'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_attention_matrix(self,
                                    attentions: torch.Tensor,
                                    layer: int = -1,
                                    head: int = 0) -> np.ndarray:
        """
        Extract specific layer and head from attention tensor

        Args:
            attentions: Attention tensor of shape (layers, heads, seq_len, seq_len)
            layer: Which layer to visualize (-1 for last layer)
            head: Which attention head to visualize

        Returns:
            Attention matrix as numpy array
        """
        if attentions is None:
            return None

        # Ensure it's on CPU and convert to numpy
        if isinstance(attentions, torch.Tensor):
            attentions = attentions.detach().cpu()

        # Select layer and head
        attention_matrix = attentions[layer, head]

        return attention_matrix.numpy()

    def _truncate_tokens(self, tokens: list, max_len: int = 64) -> list:
        """
        Truncate tokens to manageable length for visualization

        Args:
            tokens: List of tokens
            max_len: Maximum number of tokens to display

        Returns:
            Truncated token list
        """
        if len(tokens) <= max_len:
            return tokens

        # Keep first max_len tokens
        return tokens[:max_len]

    def _truncate_attention(self, attention: np.ndarray, max_len: int = 64) -> np.ndarray:
        """Truncate attention matrix to max_len x max_len"""
        if attention.shape[0] <= max_len:
            return attention

        return attention[:max_len, :max_len]

    def visualize_single(self,
                          tokens: list,
                          attentions: torch.Tensor,
                          title: str = "Attention Heatmap",
                          layer: int = -1,
                          head: int = 0,
                          max_tokens: int = 64,
                          save_name: str = "attention_single.pdf") -> str:
        """
        Visualize a single attention matrix

        Args:
            tokens: List of tokens
            attentions: Attention weights
            title: Title for the visualization
            layer: Which layer to visualize
            head: Which head to visualize
            max_tokens: Maximum tokens to display
            save_name: Output filename

        Returns:
            Path to saved PDF
        """
        # Prepare attention matrix
        attention_matrix = self._prepare_attention_matrix(attentions, layer, head)
        if attention_matrix is None:
            print("Warning: No attention weights available")
            return None

        # Truncate for visualization
        tokens_truncated = self._truncate_tokens(tokens, max_tokens)
        attention_truncated = self._truncate_attention(attention_matrix, max_tokens)

        # Save path
        save_path = self.output_dir / save_name

        # Visualize
        try:
            visualize_attention_self_attention(
                attentions=torch.tensor(attention_truncated).unsqueeze(0).unsqueeze(0).unsqueeze(0),  # Add batch, layer, head dims
                tokens=tokens_truncated,
                layer=0,
                head=0,
                top_n=5,
                mode='self_attention',
                save_path=str(save_path),
                plot_titles=[title]
            )
            print(f"✓ Saved visualization to: {save_path}")
            return str(save_path)
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None

    def visualize_decode_attention(self,
                                    tokens: list,
                                    attentions: torch.Tensor,
                                    token_type_ids: list,
                                    layer: int = -1,
                                    head: int = 0,
                                    save_name: str = "decode_attention.pdf",
                                    title: str = "Decode Stage Attention") -> str:
        """
        Visualize attention from generated tokens (decode stage)
        Shows how generated tokens attend to input tokens (including RAG context)

        Args:
            tokens: All tokens (prompt + generated)
            attentions: Full attention tensor
            token_type_ids: List indicating token type ('prompt', 'context', 'generated')
            layer: Which layer to visualize
            head: Which head to visualize
            save_name: Output filename
            title: Visualization title

        Returns:
            Path to saved PDF
        """
        if attentions is None or not tokens or not token_type_ids:
            print("Warning: Missing data for decode attention visualization")
            return None

        # Find where generated tokens start
        try:
            gen_start_idx = token_type_ids.index('generated')
        except ValueError:
            print("Warning: No generated tokens found")
            return None

        # Get attention matrix for this layer and head
        attn_matrix = self._prepare_attention_matrix(attentions, layer, head)
        if attn_matrix is None:
            return None

        # Extract attention from generated tokens to all tokens
        # Shape: [num_generated_tokens, total_seq_len]
        gen_attn = attn_matrix[gen_start_idx:, :]

        # Limit to reasonable size for visualization
        max_gen_tokens = 20  # Show last 20 generated tokens
        max_input_tokens = 180  # Show up to 180 input tokens (3x original for better RAG context visibility)

        if gen_attn.shape[0] > max_gen_tokens:
            # Take last N generated tokens
            gen_attn = gen_attn[-max_gen_tokens:, :]
            gen_tokens = tokens[gen_start_idx:][-max_gen_tokens:]
            gen_token_types = token_type_ids[gen_start_idx:][-max_gen_tokens:]
        else:
            gen_tokens = tokens[gen_start_idx:]
            gen_token_types = token_type_ids[gen_start_idx:]

        # Truncate input tokens if needed
        if gen_start_idx > max_input_tokens:
            # Keep some prompt tokens and all context tokens
            context_indices = [i for i, t in enumerate(token_type_ids[:gen_start_idx]) if t == 'context']
            if context_indices:
                # Keep tokens around context (30 tokens before context for better visibility)
                context_start = max(0, min(context_indices) - 30)
                input_tokens = tokens[context_start:gen_start_idx]
                input_token_types = token_type_ids[context_start:gen_start_idx]
                gen_attn = gen_attn[:, context_start:gen_start_idx]
            else:
                # Just truncate from end
                input_tokens = tokens[max(0, gen_start_idx - max_input_tokens):gen_start_idx]
                input_token_types = token_type_ids[max(0, gen_start_idx - max_input_tokens):gen_start_idx]
                gen_attn = gen_attn[:, max(0, gen_start_idx - max_input_tokens):gen_start_idx]
        else:
            input_tokens = tokens[:gen_start_idx]
            input_token_types = token_type_ids[:gen_start_idx]
            gen_attn = gen_attn[:, :gen_start_idx]

        # Add labels to indicate token types
        labeled_input_tokens = []
        for tok, typ in zip(input_tokens, input_token_types):
            if typ == 'context':
                labeled_input_tokens.append(f"[C]{tok}")  # Mark context tokens
            else:
                labeled_input_tokens.append(tok)

        labeled_gen_tokens = [f"[G]{tok}" for tok in gen_tokens]  # Mark generated tokens

        print(f"  Visualizing decode attention: {len(labeled_gen_tokens)} generated tokens attending to {len(labeled_input_tokens)} input tokens")

        # Create visualization using matplotlib
        save_path = self.output_dir / save_name

        try:
            import matplotlib.pyplot as plt

            # Adjust figure size based on number of tokens
            # Wider figure for more input tokens, taller for more generated tokens
            fig_width = min(30, max(14, len(labeled_input_tokens) * 0.15))
            fig_height = min(12, max(8, len(labeled_gen_tokens) * 0.4))
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Plot heatmap
            im = ax.imshow(gen_attn, aspect='auto', cmap='viridis')

            # Set labels with adaptive font size
            fontsize_x = max(5, min(8, 1200 // len(labeled_input_tokens)))  # Smaller font for more tokens
            fontsize_y = max(6, min(8, 160 // len(labeled_gen_tokens)))

            ax.set_xticks(range(len(labeled_input_tokens)))
            ax.set_yticks(range(len(labeled_gen_tokens)))
            ax.set_xticklabels(labeled_input_tokens, rotation=90, fontsize=fontsize_x)
            ax.set_yticklabels(labeled_gen_tokens, fontsize=fontsize_y)

            # Labels and title
            ax.set_xlabel('Input Tokens (Source)', fontsize=10)
            ax.set_ylabel('Generated Tokens (Query)', fontsize=10)
            ax.set_title(f"{title}\nLayer {layer}, Head {head}\n[C]=Context, [G]=Generated", fontsize=12)

            # Colorbar
            plt.colorbar(im, ax=ax, label='Attention Weight')

            # Tight layout
            plt.tight_layout()

            # Save
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()

            print(f"✓ Saved decode attention visualization to: {save_path}")
            return str(save_path)

        except Exception as e:
            print(f"Error creating decode attention visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def visualize_comparison(self,
                              tokens_no_rag: list,
                              attentions_no_rag: torch.Tensor,
                              tokens_rag: list,
                              attentions_rag: torch.Tensor,
                              layer: int = -1,
                              head: int = 0,
                              max_tokens: int = 48,
                              save_name: str = "attention_comparison.pdf",
                              query: str = "",
                              token_type_ids_no_rag: list = None,
                              token_type_ids_rag: list = None) -> dict:
        """
        Create comparison visualization between RAG and non-RAG attention

        Args:
            tokens_no_rag: Tokens from non-RAG generation
            attentions_no_rag: Attention weights from non-RAG
            tokens_rag: Tokens from RAG generation
            attentions_rag: Attention weights from RAG
            layer: Which layer to compare
            head: Which head to compare
            max_tokens: Maximum tokens to display
            save_name: Output filename
            query: The query being answered

        Returns:
            dict with paths to saved visualizations
        """
        results = {}

        # Check tokens are not None or empty
        if not tokens_no_rag or not tokens_rag:
            print("Warning: Token lists are empty or None")
            return results

        # Prepare attention matrices
        attn_no_rag = self._prepare_attention_matrix(attentions_no_rag, layer, head)
        attn_rag = self._prepare_attention_matrix(attentions_rag, layer, head)

        if attn_no_rag is None or attn_rag is None:
            print("Warning: Attention weights not available for comparison")
            print(f"  attn_no_rag is None: {attn_no_rag is None}")
            print(f"  attn_rag is None: {attn_rag is None}")
            return results

        # For comparison, we need to use common tokens or truncate to same length
        # We'll use the shorter length
        min_len = min(len(tokens_no_rag), len(tokens_rag), max_tokens)
        print(f"  Token lengths: no_rag={len(tokens_no_rag)}, rag={len(tokens_rag)}, using min_len={min_len}")

        tokens_no_rag_truncated = tokens_no_rag[:min_len]
        tokens_rag_truncated = tokens_rag[:min_len]

        attn_no_rag_truncated = attn_no_rag[:min_len, :min_len]
        attn_rag_truncated = attn_rag[:min_len, :min_len]

        # Create individual visualizations
        print("\nCreating individual attention visualizations...")

        # No RAG visualization
        no_rag_path = self.output_dir / f"no_rag_{save_name}"
        self.visualize_single(
            tokens_no_rag,
            attentions_no_rag,
            title=f"Without RAG - Layer {layer}, Head {head}",
            layer=layer,
            head=head,
            max_tokens=max_tokens,
            save_name=f"no_rag_{save_name}"
        )
        results['no_rag_path'] = str(no_rag_path)

        # RAG visualization
        rag_path = self.output_dir / f"with_rag_{save_name}"
        self.visualize_single(
            tokens_rag,
            attentions_rag,
            title=f"With RAG - Layer {layer}, Head {head}",
            layer=layer,
            head=head,
            max_tokens=max_tokens,
            save_name=f"with_rag_{save_name}"
        )
        results['rag_path'] = str(rag_path)

        # Create comparison visualization
        print("\nCreating comparison heatmap...")
        comparison_path = self.output_dir / f"comparison_{save_name}"

        try:
            # Use IzzyViz's comparison function
            compare_two_attentions_with_circles(
                attn1=torch.tensor(attn_no_rag_truncated),
                attn2=torch.tensor(attn_rag_truncated),
                tokens=tokens_no_rag_truncated,  # Use one set of tokens
                title=f"Comparison: Without RAG vs With RAG\nLayer {layer}, Head {head}",
                save_path=str(comparison_path),
                circle_scale=1.2
            )
            results['comparison_path'] = str(comparison_path)
            print(f"✓ Saved comparison to: {comparison_path}")
        except Exception as e:
            print(f"Error creating comparison visualization: {e}")
            import traceback
            traceback.print_exc()

        # Create decode-stage attention visualizations (showing generated tokens attending to input)
        print("\nCreating decode-stage attention visualizations...")

        # Decode attention for no-RAG
        if token_type_ids_no_rag:
            decode_no_rag_path = self.output_dir / f"decode_no_rag_{save_name}"
            decode_path = self.visualize_decode_attention(
                tokens=tokens_no_rag,
                attentions=attentions_no_rag,
                token_type_ids=token_type_ids_no_rag,
                layer=layer,
                head=head,
                save_name=f"decode_no_rag_{save_name}",
                title="Decode Attention - Without RAG"
            )
            if decode_path:
                results['decode_no_rag_path'] = decode_path

        # Decode attention for RAG
        if token_type_ids_rag:
            decode_rag_path = self.output_dir / f"decode_with_rag_{save_name}"
            decode_path = self.visualize_decode_attention(
                tokens=tokens_rag,
                attentions=attentions_rag,
                token_type_ids=token_type_ids_rag,
                layer=layer,
                head=head,
                save_name=f"decode_with_rag_{save_name}",
                title="Decode Attention - With RAG (showing attention to context)"
            )
            if decode_path:
                results['decode_rag_path'] = decode_path

        return results

    def batch_visualize_heads(self,
                               tokens: list,
                               attentions: torch.Tensor,
                               layer: int = -1,
                               num_heads: int = 4,
                               prefix: str = "head",
                               max_tokens: int = 48) -> list:
        """
        Visualize multiple attention heads

        Args:
            tokens: Token list
            attentions: Attention tensor
            layer: Which layer
            num_heads: How many heads to visualize
            prefix: Filename prefix
            max_tokens: Max tokens to show

        Returns:
            List of saved file paths
        """
        saved_paths = []

        if attentions is None:
            return saved_paths

        total_heads = attentions.shape[1] if len(attentions.shape) > 1 else 1

        for head in range(min(num_heads, total_heads)):
            save_name = f"{prefix}_layer{layer}_head{head}.pdf"
            path = self.visualize_single(
                tokens=tokens,
                attentions=attentions,
                title=f"Layer {layer}, Head {head}",
                layer=layer,
                head=head,
                max_tokens=max_tokens,
                save_name=save_name
            )
            if path:
                saved_paths.append(path)

        return saved_paths


if __name__ == '__main__':
    # Test attention visualizer
    print("Testing Attention Visualizer...")

    # Create dummy attention data for testing
    seq_len = 20
    num_layers = 2
    num_heads = 4

    # Random attention matrices
    dummy_attentions = torch.randn(num_layers, num_heads, seq_len, seq_len)
    dummy_attentions = torch.softmax(dummy_attentions, dim=-1)

    dummy_tokens = [f"token_{i}" for i in range(seq_len)]

    visualizer = AttentionVisualizer()

    # Test single visualization
    print("\n1. Testing single visualization...")
    visualizer.visualize_single(
        tokens=dummy_tokens,
        attentions=dummy_attentions,
        title="Test Attention",
        layer=-1,
        head=0,
        save_name="test_single.pdf"
    )

    # Test comparison
    print("\n2. Testing comparison visualization...")
    dummy_attentions_2 = torch.randn(num_layers, num_heads, seq_len, seq_len)
    dummy_attentions_2 = torch.softmax(dummy_attentions_2, dim=-1)

    visualizer.visualize_comparison(
        tokens_no_rag=dummy_tokens,
        attentions_no_rag=dummy_attentions,
        tokens_rag=dummy_tokens,
        attentions_rag=dummy_attentions_2,
        layer=-1,
        head=0,
        save_name="test_comparison.pdf"
    )

    print("\n✓ Testing complete!")
