import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = ["Existe alguma ação onde o locátorio foi multado?", "Existe alguma ação trabalhista que aborde insalubridade?", "Cite uma jurisprudência que possua danos morais?"];

const GPT4V_EXAMPLES: string[] = [
    "Me de 3 exemplos de causas trabalhistas?",
    "Existe alguma apelação Cível Contrato de Locação de Imóvel?",
    "Faça um resumo da ação de registro 2024.0000380427"
];

interface Props {
    onExampleClicked: (value: string) => void;
    useGPT4V?: boolean;
}

export const ExampleList = ({ onExampleClicked, useGPT4V }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {(useGPT4V ? GPT4V_EXAMPLES : DEFAULT_EXAMPLES).map((question, i) => (
                <li key={i}>
                    <Example text={question} value={question} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
